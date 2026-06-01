"""
Standby audio music detector — finishes the auto-standby feature the user
explicitly asked for in bobert_memory.json: 'standby mode trigger condition
update', a.k.a. "lyrics/song audio detection, not simply when a play command
is issued".

The existing ambient-music path in bobert_companion only fires on Whisper's
text-side markers ([Music], (Music), ♪ etc.), which Whisper emits
inconsistently and never for vocal music with intelligible lyrics. This
skill adds two complementary detection paths:

  1. Spectral classifier (pure-numpy FFT, no extra deps) — flags sustained
     tonal/harmonic content. Drives is_music_currently_playing() and the
     wake-word gating in should_refuse_wake().

  2. Background lyric-detection loop — runs whisper-tiny on a 3-second
     rolling mic buffer every 5 seconds, scores the transcript for
     rhyme-density + onset-energy (librosa), and after 15 consecutive
     seconds of positive detection AND the headset being the active
     output auto-engages standby/wake-word-only mode and TTS
     "I'll wait until you call, sir". Disabled silently if librosa isn't
     installed — the spectral detector keeps working either way.

Heuristic per chunk (spectral classifier, pure-numpy FFT, no extra deps):
  - Spectral flatness < FLATNESS_THRESHOLD  (low flatness = tonal/harmonic)
  - Bass-band (60-250 Hz) energy ratio > BASS_RATIO_THRESHOLD
  - RMS above MIN_RMS (skip near-silence)

Sustained-music state is set when ≥ MUSIC_FRACTION_REQUIRED of the audio
fed in the last WINDOW_SECONDS classifies as musical. While that state is
active, should_refuse_wake() returns True for wake-word utterances that
look like lyric near-misses.

Public API (looked up by bobert_companion via sys.modules at runtime):
    feed_audio(audio, sample_rate)        — call after each record_speech()
    is_music_currently_playing() -> bool  — True once sustained ≥ 15 s
    should_refuse_wake(text)     -> bool  — gate for the wake-word check
    music_state_summary()        -> str   — used by the manual action
"""

import sys
import threading
import time
from collections import deque

import numpy as np


# ── tunables ─────────────────────────────────────────────────────────────
FLATNESS_THRESHOLD       = 0.25     # spectrum below this → tonal/musical
BASS_RATIO_THRESHOLD     = 0.10     # ≥ 10 % energy in 60-250 Hz band
MIN_RMS                  = 0.005    # ignore near-silent chunks
WINDOW_SECONDS           = 15.0     # sliding window for sustained-music check
MIN_WINDOW_COVERAGE_SEC  = 5.0      # need ≥ this much audio in window before deciding
MUSIC_FRACTION_REQUIRED  = 0.75     # ≥ 75 % of recent audio musical → active
MUSIC_TIMEOUT_SECONDS    = 30.0     # if no audio fed for this long → reset
SUSTAINED_HOLD_SECONDS   = 15.0     # must be active this long before gating wakes
NEAR_MISS_MAX_WORDS      = 3        # ≤ this many words = "clear" wake utterance
# Background-loop tunables imported from core/config at register() time so
# the loop reads the live values instead of frozen module-load constants.
_LOOP_DEFAULTS = {
    "enabled":           True,
    "buffer_seconds":    3.0,
    "check_interval":    5.0,
    "match_windows":     3,
    "onset_min":         0.30,
    "rhyme_min":         0.30,
    "whisper_model":     "tiny",
}
# ─────────────────────────────────────────────────────────────────────────


_lock = threading.Lock()
_classifications: "deque[tuple[float, bool, float]]" = deque()   # (ts, is_musical, duration_sec)
_music_active = [False]
_music_since  = [0.0]
_last_feed_at = [0.0]
_total_chunks_seen  = [0]
_total_music_chunks = [0]

# ── background-loop state ────────────────────────────────────────────────
_loop_stop = threading.Event()
_loop_thread: "threading.Thread | None" = None
_loop_cfg: dict = dict(_LOOP_DEFAULTS)
_loop_consecutive = [0]                 # consecutive matched windows
_whisper_model = [None]                 # lazy-loaded whisper-tiny handle
_librosa_mod = [None]                   # cached librosa module (None=unavailable)
_loop_last_score: dict = {"ts": 0.0, "onset": 0.0, "rhyme": 0.0, "text": ""}


def _classify_chunk(audio: np.ndarray, sample_rate: int) -> bool:
    """Spectral classifier — True if the chunk looks like sustained
    tonal/harmonic content (music) rather than speech or noise."""
    if audio is None or audio.size < sample_rate // 4:   # need ≥ 0.25 s
        return False
    a = audio
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32, copy=False)

    rms = float(np.sqrt(np.mean(a * a)))
    if rms < MIN_RMS:
        return False

    n = 1 << max(8, (a.size - 1).bit_length())
    spec = np.abs(np.fft.rfft(a, n=n))
    psd = spec * spec + 1e-12

    log_mean   = float(np.exp(np.mean(np.log(psd))))
    arith_mean = float(np.mean(psd))
    flatness   = log_mean / max(arith_mean, 1e-12)

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    bass_mask    = (freqs >= 60.0) & (freqs <= 250.0)
    bass_energy  = float(psd[bass_mask].sum())
    total_energy = float(psd.sum())
    bass_ratio   = bass_energy / max(total_energy, 1e-12)

    return flatness < FLATNESS_THRESHOLD and bass_ratio > BASS_RATIO_THRESHOLD


def feed_audio(audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Classify one chunk of recorded audio and update rolling state.
    bobert_companion calls this after every successful record_speech()."""
    try:
        if audio is None:
            return
        is_musical = _classify_chunk(audio, sample_rate)
    except Exception:
        return

    duration = float(audio.size) / float(max(sample_rate, 1))
    now = time.time()

    with _lock:
        _last_feed_at[0] = now
        _classifications.append((now, is_musical, duration))
        cutoff = now - WINDOW_SECONDS
        while _classifications and _classifications[0][0] < cutoff:
            _classifications.popleft()

        _total_chunks_seen[0] += 1
        if is_musical:
            _total_music_chunks[0] += 1

        total_dur = sum(d for _, _, d in _classifications)
        music_dur = sum(d for _, m, d in _classifications if m)

        if (total_dur >= MIN_WINDOW_COVERAGE_SEC
                and (music_dur / max(total_dur, 1e-9)) >= MUSIC_FRACTION_REQUIRED):
            if not _music_active[0]:
                _music_active[0] = True
                _music_since[0]  = now
                print(f"  [audio-music] sustained music detected "
                      f"({music_dur:.1f}/{total_dur:.1f}s in window)")
        else:
            if _music_active[0]:
                _music_active[0] = False
                print("  [audio-music] music ended")


def _reset_if_stale() -> None:
    """Clear music state if we haven't been fed audio for a while
    (caller already holds _lock)."""
    if not _music_active[0]:
        return
    if (time.time() - _last_feed_at[0]) > MUSIC_TIMEOUT_SECONDS:
        _music_active[0] = False
        _classifications.clear()
        print("  [audio-music] state expired (no audio fed)")


def is_music_currently_playing() -> bool:
    """True only after the spectral classifier has flagged sustained music
    for at least SUSTAINED_HOLD_SECONDS — gives the >15 s requirement the
    spec calls for."""
    with _lock:
        _reset_if_stale()
        if not _music_active[0]:
            return False
        return (time.time() - _music_since[0]) >= SUSTAINED_HOLD_SECONDS


def should_refuse_wake(text: str) -> bool:
    """Decide whether the standby loop should ignore a wake-word match.
    Returns True if the env is musical AND the utterance looks like a
    lyric near-miss rather than a clear standalone wake command."""
    if not is_music_currently_playing():
        return False
    if not text:
        return True
    words = text.strip().lower().split()
    if not words:
        return True
    if len(words) <= NEAR_MISS_MAX_WORDS:
        first = words[0].strip(",.!?")
        # A clear wake looks like 'jarvis', 'hey jarvis', 'ok jarvis' — short,
        # wake word at the front. Anything longer or buried mid-sentence
        # while music is playing is treated as a lyric and refused.
        if first in {"jarvis", "hey", "ok", "okay"}:
            return False
    return True


def music_state_summary() -> str:
    with _lock:
        _reset_if_stale()
        if _music_active[0]:
            dur = time.time() - _music_since[0]
            if dur >= SUSTAINED_HOLD_SECONDS:
                return (f"Music has been playing for {dur:.0f} seconds, sir. "
                        f"Wake-word filtering is active — only a clear 'JARVIS' "
                        f"will bring me back.")
            return (f"Tonal audio detected {dur:.0f} seconds ago, sir. "
                    f"Still confirming.")
        seen = _total_chunks_seen[0]
        if seen == 0:
            return "No audio analysed yet, sir."
        pct = 100.0 * _total_music_chunks[0] / seen
        return (f"No sustained music at the moment, sir. "
                f"{_total_music_chunks[0]} of {seen} chunks classified as "
                f"musical ({pct:.0f}%) since startup.")


# ── background lyric-detection loop ──────────────────────────────────────

_WHISPER_MUSIC_MARKERS = ("[music]", "(music)", "♪", "[singing]",
                          "(singing)", "[applause]", "[laughter]")


def _try_import_librosa():
    """Cache-and-return the librosa module, or None if it isn't installed.
    Imported lazily so JARVIS still boots cleanly when librosa isn't
    available — the loop then disables itself with a clear console note."""
    if _librosa_mod[0] is not None:
        return _librosa_mod[0]
    try:
        import librosa as _l  # noqa: F401
        _librosa_mod[0] = _l
        return _l
    except Exception:
        return None


def _ensure_whisper_tiny():
    """Lazy-load whisper-tiny ONCE for use by the background loop. Prefers
    faster-whisper (already a JARVIS dep) and falls back to openai-whisper.
    Returns the model or None on failure.

    GPU: faster-whisper rides ctranslate2's OWN CUDA runtime (NOT torch), so
    this loop can run on the 3090 even though the rest of the torch stack is
    CPU-only on Python 3.14. The cublas/cudnn DLLs are already added to PATH at
    boot by bobert_companion._register_cuda_dll_dirs(). We try CUDA first
    (float16, ~10x faster + frees a CPU core that face-tracking needs) and fall
    back to CPU/int8 if the GPU path fails. Flipped from hardcoded CPU in the
    2026-05-30 GPU audit.
    """
    if _whisper_model[0] is not None:
        return _whisper_model[0]
    model_name = _loop_cfg.get("whisper_model", "tiny")
    try:
        from faster_whisper import WhisperModel as _FWM
        # Try GPU first — same ctranslate2 CUDA path the main STT uses.
        try:
            _whisper_model[0] = _FWM(model_name, device="cuda", compute_type="float16")
            print(f"  [standby-loop] faster-whisper '{model_name}' ready on cuda")
            return _whisper_model[0]
        except Exception as e_gpu:
            print(f"  [standby-loop] faster-whisper cuda load failed "
                  f"({type(e_gpu).__name__}); falling back to cpu/int8")
        # CPU fallback — int8 keeps it light on the desk CPU.
        try:
            _whisper_model[0] = _FWM(model_name, device="cpu", compute_type="int8")
            print(f"  [standby-loop] faster-whisper '{model_name}' ready on cpu")
            return _whisper_model[0]
        except Exception as e:
            print(f"  [standby-loop] faster-whisper load failed: {e}")
    except ImportError:
        pass
    try:
        # openai-whisper fallback uses torch — stays CPU until a CUDA-enabled
        # torch wheel exists for Python 3.14. Device-aware so it auto-upgrades
        # the moment torch.cuda becomes available.
        import whisper as _wlib
        _dev = "cpu"
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _dev = "cuda"
        except Exception:
            pass
        _whisper_model[0] = _wlib.load_model(model_name, device=_dev)
        print(f"  [standby-loop] openai-whisper '{model_name}' ready on {_dev}")
        return _whisper_model[0]
    except Exception as e:
        print(f"  [standby-loop] whisper load failed: {e}")
        return None


def _transcribe_buffer(audio: np.ndarray, sample_rate: int) -> str:
    """Run whisper-tiny on the rolling buffer and return the transcript
    text (lowercased), or "" on failure. Handles both faster-whisper and
    openai-whisper return shapes."""
    model = _ensure_whisper_tiny()
    if model is None or audio is None or audio.size == 0:
        return ""
    a = audio.astype(np.float32, copy=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    try:
        if hasattr(model, "transcribe"):
            # Pin language to English: the user is monolingual English, and
            # whisper-tiny otherwise auto-detects per chunk and hallucinates
            # foreign-language tokens ('nn'/'cy'/'ko') on ~16% of short
            # noisy buffers, polluting lyric detection. Both faster-whisper
            # (WhisperModel.transcribe) and openai-whisper (load_model().
            # transcribe) accept the same language= kwarg, so this single
            # call covers both engine paths.
            out = model.transcribe(a, language="en")
            if isinstance(out, tuple):
                segs, _info = out
                text = " ".join(getattr(s, "text", "") for s in segs)
            elif isinstance(out, dict):
                text = out.get("text", "")
            else:
                text = str(out)
            return (text or "").strip().lower()
    except Exception as e:
        print(f"  [standby-loop] transcribe failed: {e}")
    return ""


def _onset_energy(audio: np.ndarray, sample_rate: int) -> float:
    """Mean librosa onset-strength (spectral flux) for the buffer.
    Returns 0.0 if librosa isn't available or computation fails."""
    lib = _try_import_librosa()
    if lib is None or audio is None or audio.size == 0:
        return 0.0
    a = audio.astype(np.float32, copy=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    try:
        env = lib.onset.onset_strength(y=a, sr=sample_rate)
        if env is None or env.size == 0:
            return 0.0
        return float(np.mean(env))
    except Exception:
        return 0.0


def _rhyme_density(text: str) -> float:
    """Cheap rhyme heuristic: share of adjacent word pairs that share the
    final 2-character phonetic-ish suffix. Linguistically crude but enough
    to separate sung lyric structure from conversational prose."""
    if not text:
        return 0.0
    words = [w.strip(",.!?;:\"'()[]") for w in text.split() if w.strip()]
    words = [w for w in words if len(w) >= 3 and w.isalpha()]
    if len(words) < 4:
        return 0.0
    suffixes = [w[-2:].lower() for w in words]
    matches = sum(1 for i in range(1, len(suffixes))
                  if suffixes[i] == suffixes[i - 1])
    return matches / float(len(suffixes) - 1)


def _looks_like_lyrics(text: str, onset: float) -> bool:
    """Combine the three signals into a single per-window verdict."""
    if not text and onset < _loop_cfg["onset_min"]:
        return False
    if any(m in text for m in _WHISPER_MUSIC_MARKERS):
        return True
    rhyme = _rhyme_density(text)
    _loop_last_score.update({"ts": time.time(), "onset": onset,
                             "rhyme": rhyme, "text": text})
    musical_audio = onset >= _loop_cfg["onset_min"]
    rhyme_dense   = rhyme >= _loop_cfg["rhyme_min"]
    return musical_audio and rhyme_dense


def _suppress_due_to_state(bc) -> bool:
    """Don't trip auto-standby if JARVIS is already dormant or just played
    music itself (the player audio is the source bleeding into the mic)."""
    try:
        if getattr(bc, "_standby_mode")[0] or getattr(bc, "_sleep_mode")[0]:
            return True
    except Exception:
        pass
    try:
        last_played = getattr(bc, "_jarvis_played_music_at")[0]
        if last_played and (time.time() - last_played) < 60.0:
            return True
    except Exception:
        pass
    return False


def _background_loop() -> None:
    """Daemon loop: capture → transcribe → score → maybe engage standby."""
    print("  [standby-loop] background lyric-detection loop started")
    interval = float(_loop_cfg["check_interval"])
    buffer_s = float(_loop_cfg["buffer_seconds"])
    threshold = int(_loop_cfg["match_windows"])

    while not _loop_stop.wait(interval):
        bc = sys.modules.get("bobert_companion")
        if bc is None:
            continue
        if _suppress_due_to_state(bc):
            _loop_consecutive[0] = 0
            continue
        try:
            sample_rate = int(getattr(bc, "SAMPLE_RATE", 16000))
            audio = bc.get_mic_buffer(buffer_s, sample_rate=sample_rate)
        except Exception as e:
            print(f"  [standby-loop] mic buffer failed: {e}")
            continue
        if audio is None or audio.size == 0:
            _loop_consecutive[0] = 0
            continue

        try:
            text  = _transcribe_buffer(audio, sample_rate)
            onset = _onset_energy(audio, sample_rate)
            is_lyric = _looks_like_lyrics(text, onset)
        except Exception as e:
            print(f"  [standby-loop] scoring failed: {e}")
            # Release the buffer explicitly — leaving a 3s float32 array
            # pinned per failed iteration would bloat RSS over hours of
            # uptime.
            audio = None
            continue
        # explicit release; np arrays are tiny but the loop runs forever
        audio = None

        if is_lyric:
            _loop_consecutive[0] += 1
            if _loop_consecutive[0] >= threshold:
                using_headset = False
                try:
                    using_headset = bool(bc.is_using_headset())
                except Exception:
                    pass
                if not using_headset:
                    # require headset so we don't auto-standby because the
                    # room speakers are playing — only the personal headset
                    # context is unambiguous enough for the spec.
                    continue
                engage = getattr(bc, "_standby_auto_engage", None)
                if engage is None:
                    print("  [standby-loop] bridge function missing — cannot engage")
                    _loop_consecutive[0] = 0
                    continue
                try:
                    fired = bool(engage("vocal-music"))
                except Exception as e:
                    print(f"  [standby-loop] engage failed: {e}")
                    fired = False
                if fired:
                    print(f"  [standby-loop] standby engaged after "
                          f"{_loop_consecutive[0]} consecutive lyric windows "
                          f"(onset={_loop_last_score['onset']:.2f}, "
                          f"rhyme={_loop_last_score['rhyme']:.2f})")
                _loop_consecutive[0] = 0
        else:
            _loop_consecutive[0] = 0
    print("  [standby-loop] background lyric-detection loop stopped")


def _load_loop_cfg() -> None:
    """Pull the live STANDBY_LOOP_* values from core.config into _loop_cfg.
    Falls back to _LOOP_DEFAULTS for any name that isn't defined yet."""
    try:
        from core import config as _cfg
    except Exception:
        return
    mapping = {
        "enabled":        "STANDBY_LOOP_ENABLED",
        "buffer_seconds": "STANDBY_LOOP_BUFFER_SECONDS",
        "check_interval": "STANDBY_LOOP_CHECK_INTERVAL_SEC",
        "match_windows":  "STANDBY_LOOP_MATCH_WINDOWS",
        "onset_min":      "STANDBY_LOOP_ONSET_ENERGY_MIN",
        "rhyme_min":      "STANDBY_LOOP_RHYME_RATIO_MIN",
        "whisper_model":  "STANDBY_LOOP_WHISPER_MODEL",
    }
    for k, attr in mapping.items():
        if hasattr(_cfg, attr):
            _loop_cfg[k] = getattr(_cfg, attr)


def _start_background_loop() -> None:
    """Start the daemon thread if config allows AND librosa is available.
    Idempotent — no-op if already running."""
    global _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        return
    if not _loop_cfg.get("enabled", True):
        print("  [standby-loop] disabled via STANDBY_LOOP_ENABLED")
        return
    if _try_import_librosa() is None:
        print("  [standby-loop] librosa not installed — auto-standby "
              "lyric detection disabled (pip install librosa)")
        return
    _loop_stop.clear()
    _loop_thread = threading.Thread(
        target=_background_loop,
        name="standby-audio-loop",
        daemon=True,
    )
    _loop_thread.start()


def stop_background_loop() -> None:
    """Signal the daemon thread to exit. Called from bobert_companion's
    shutdown path if/when wired in; safe to call even if never started."""
    _loop_stop.set()


def register(actions):
    def audio_music_status(_: str = "") -> str:
        try:
            return music_state_summary()
        except Exception as e:
            return f"audio music status failed: {e}"

    actions["audio_music_status"] = audio_music_status

    _load_loop_cfg()
    _start_background_loop()
    print("  [audio-music] spectral detector loaded — feed_audio() ready")
