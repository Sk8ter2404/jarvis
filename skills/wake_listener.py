"""
Background wake-word listener.

Runs a `core.wake_word.WakeWordDetector` on a dedicated thread for the
lifetime of the JARVIS session. When the detector hears a wake phrase
("hey jarvis", "jarvis"), the `_on_detect` callback prints a log line
and — if the main loop is in sleep mode — calls
`bobert_companion.proactive_announce` so the next turn boundary speaks
an acknowledgement. No event file is persisted; the in-process callback
is the only channel.

Voice-biometric gating
----------------------
After the user enrolls their voice via skills/enroll_voice.py, the wake
event is gated on a speaker-ID check: the most recent ~2 s of audio
(pre + post wake-word) is fed to `core.voice_id.identify_speaker` and the
proactive announce only fires when the cosine similarity beats the
configured threshold (default 0.72). Ambient TV/music wake-ups that
don't match the user's voiceprint are rejected. The check is a permissive
no-op when resemblyzer is missing, no speakers are enrolled, or guest
mode is enabled — that preserves backward compatibility for fresh
installs.

Config knobs (set as module-level constants in this file; they can
also be overridden via the `wake_listener_configure` action so the
voice-control path can adjust them without a restart):

    WAKE_WORDS         — list of phrases the detector should watch for
    WAKE_WORD_ENGINE   — 'openwakeword' | 'porcupine' | 'off'
    WAKE_WORD_THRESHOLD — 0..1, detection sensitivity
    WAKE_WORD_DEVICE    — sounddevice input index, or None to auto
    VOICE_BIOMETRIC_ENABLED  — gate wake events on speaker ID
    VOICE_BIOMETRIC_THRESHOLD — cosine-similarity floor for accepting a match
    GUEST_MODE_ENABLED        — temporary bypass for visitors

Registered actions
------------------
    wake_listener_start    — start the detector (idempotent)
    wake_listener_stop     — stop the detector
    wake_listener_status   — short human-readable status string
    wake_listener_configure — change runtime config; arg is "key=value"
    guest_mode_on / guest_mode_off — toggle gating bypass for visitors
    voice_gating_on / voice_gating_off — toggle biometric gate entirely

If `openwakeword` / `pvporcupine` aren't installed the skill loads
cleanly and prints a friendly install hint when start is attempted —
JARVIS continues to operate on hotkey + VAD as before.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np


# ── tunables (overridable via wake_listener_configure) ───────────────
WAKE_WORDS: list[str] = ["hey jarvis", "jarvis"]
WAKE_WORD_ENGINE: str = "openwakeword"
WAKE_WORD_THRESHOLD: float = 0.5
WAKE_WORD_DEVICE: Optional[int] = None
WAKE_WORD_USE_SILERO_VAD: bool = False
# Autostart disabled by default: the detector opens its own sd.InputStream
# on the same device that bobert_companion.record_speech() uses, and Windows
# WASAPI cannot share an input device between two streams (raises
# PortAudioError). User must opt in explicitly via wake_listener_start
# (e.g. "jarvis, start listening for the wake word").
WAKE_WORD_AUTOSTART: bool = False

# Voice biometric gating — verify the post-wake utterance matches the user's
# enrolled voiceprint before triggering the proactive announce. Default
# is ON: when no speakers are enrolled or resemblyzer is missing the gate
# falls back to permissive (no rejection), so legacy single-user installs
# keep working. bobert_companion.py may override these at register-time.
VOICE_BIOMETRIC_ENABLED: bool = True
VOICE_BIOMETRIC_THRESHOLD: float = 0.72
# Seconds of audio to capture pre/post wake-word for the speaker ID embed.
# Resemblyzer's MIN_IDENTIFY_SECONDS is 0.6; we collect 1.5 s before + 1.0 s
# after so the embed covers the wake utterance itself plus any trailing
# command speech.
VOICE_BIOMETRIC_PRE_SECONDS: float = 1.5
VOICE_BIOMETRIC_POST_SECONDS: float = 1.0
# Guest mode: temporarily bypass the gate so visitors can wake JARVIS
# without being enrolled. Stored in-memory only — resets to off when
# wake_listener is restarted, so guests have to re-enable per boot.
GUEST_MODE_ENABLED: bool = False
# ─────────────────────────────────────────────────────────────────────


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_detector = None
# RLock so _stop_voice_tap() can re-enter from wake_listener_stop(), which
# already holds _lock when calling it.
_lock = threading.RLock()

# Rolling pre-wake audio buffer fed by a tap on the detector InputStream.
# A small background thread drains the tap queue into _voice_audio_buffer
# so the per-frame PortAudio callback never has to do extra work.
_voice_id_lock = threading.Lock()
_voice_audio_tap: Optional["queue.Queue[np.ndarray]"] = None
_voice_audio_buffer: "deque[np.ndarray]" = deque()
_voice_buffer_lock = threading.Lock()
_voice_buffer_thread: Optional[threading.Thread] = None
_voice_buffer_stop: Optional[threading.Event] = None
_voice_buffer_capacity_frames: int = 0


def _ensure_core_on_path() -> None:
    """Make `from core.wake_word import WakeWordDetector` work even
    when the skill is loaded via importlib.util.spec_from_file_location
    (which does not inject the project root into sys.path)."""
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)


def _apply_bobert_overrides() -> None:
    """Pick up VOICE_BIOMETRIC_* / GUEST_MODE_ENABLED defaults from
    bobert_companion if it's already imported. Lets the user's master
    config block in bobert_companion.py act as the boot defaults for
    this skill — fresh installs that haven't enrolled a voice yet can
    leave the gate off via the top-level constant."""
    global VOICE_BIOMETRIC_ENABLED, VOICE_BIOMETRIC_THRESHOLD
    global GUEST_MODE_ENABLED
    bobert = sys.modules.get("bobert_companion") or sys.modules.get("__main__")
    if bobert is None:
        return
    val = getattr(bobert, "VOICE_BIOMETRIC_ENABLED", None)
    if isinstance(val, bool):
        VOICE_BIOMETRIC_ENABLED = val
    val = getattr(bobert, "VOICE_BIOMETRIC_THRESHOLD", None)
    if isinstance(val, (int, float)):
        with _voice_id_lock:
            VOICE_BIOMETRIC_THRESHOLD = float(val)
    val = getattr(bobert, "GUEST_MODE_ENABLED", None)
    if isinstance(val, bool):
        GUEST_MODE_ENABLED = val


def _drain_voice_tap() -> None:
    """Pull frames out of the detector's tap queue into the rolling
    deque. Runs on its own daemon thread so the PortAudio callback in
    core.wake_word never blocks behind voice-ID work."""
    while _voice_buffer_stop is not None and not _voice_buffer_stop.is_set():
        tap = _voice_audio_tap
        if tap is None:
            time.sleep(0.05)
            continue
        try:
            frame = tap.get(timeout=0.2)
        except queue.Empty:
            continue
        except Exception:
            continue
        with _voice_buffer_lock:
            _voice_audio_buffer.append(frame)
            while len(_voice_audio_buffer) > _voice_buffer_capacity_frames:
                _voice_audio_buffer.popleft()


def _start_voice_tap(det) -> None:
    """Attach a tap to the detector and start the drain thread. Safe to
    call repeatedly — second call is a no-op."""
    global _voice_audio_tap, _voice_buffer_thread, _voice_buffer_stop
    global _voice_buffer_capacity_frames
    if _voice_audio_tap is not None:
        return
    if not hasattr(det, "add_tap"):
        return
    # Capacity: 80 ms frames at 16 kHz → 12.5 frames/sec. Hold enough
    # to cover pre + post window with one second of headroom.
    frames_per_sec = 12.5
    total_secs = (VOICE_BIOMETRIC_PRE_SECONDS
                  + VOICE_BIOMETRIC_POST_SECONDS + 1.0)
    _voice_audio_tap = queue.Queue()
    with _voice_buffer_lock:
        _voice_buffer_capacity_frames = max(8, int(total_secs * frames_per_sec))
        _voice_audio_buffer.clear()
    _voice_buffer_stop = threading.Event()
    try:
        det.add_tap(_voice_audio_tap)
    except Exception as e:
        print(f"  [wake-listener] add_tap failed: {e}")
        _voice_audio_tap = None
        _voice_buffer_stop = None
        return
    _voice_buffer_thread = threading.Thread(
        target=_drain_voice_tap,
        name="wake-listener-voice-tap",
        daemon=True,
    )
    _voice_buffer_thread.start()


def _stop_voice_tap() -> None:
    """Detach the tap and stop the drain thread. Safe if not started."""
    global _voice_audio_tap, _voice_buffer_thread, _voice_buffer_stop
    # Snapshot _detector under _lock so a concurrent wake_listener_stop()
    # / wake_listener_configure() can't reassign it to None between our
    # validity check and the remove_tap() call. _lock is an RLock, so this
    # is safe even when called from wake_listener_stop() which already
    # holds it.
    with _lock:
        det = _detector
    if _voice_buffer_stop is not None:
        _voice_buffer_stop.set()
    tap = _voice_audio_tap
    if tap is not None and det is not None:
        try:
            if hasattr(det, "remove_tap"):
                det.remove_tap(tap)
        except Exception:
            pass
    _voice_audio_tap = None
    _voice_buffer_thread = None
    _voice_buffer_stop = None
    with _voice_buffer_lock:
        _voice_audio_buffer.clear()


def _snapshot_voice_audio(seconds: float, sample_rate: int) -> Optional[np.ndarray]:
    """Return the most recent `seconds` of buffered audio as a single
    float32 array, or None if the buffer is empty."""
    need = max(1, int(seconds * sample_rate))
    with _voice_buffer_lock:
        if not _voice_audio_buffer:
            return None
        frames = list(_voice_audio_buffer)
    try:
        concat = np.concatenate(frames).astype(np.float32, copy=False)
    except Exception:
        return None
    if concat.size == 0:
        return None
    return concat[-need:] if concat.size > need else concat


def _identify_recent_speaker() -> tuple[Optional[str], float]:
    """Wait briefly so post-wake frames flow into the rolling buffer,
    then run voice ID on the captured window. Returns (speaker or None,
    cosine similarity). Threshold checking happens inside core.voice_id;
    if our configured threshold differs from voice_id's default we
    temporarily swap it for the call, guarded by _voice_id_lock so a
    concurrent caller can't observe the modified value."""
    time.sleep(VOICE_BIOMETRIC_POST_SECONDS)
    sr = 16000
    det = _detector
    if det is not None:
        sr = int(getattr(det, "sample_rate", 16000) or 16000)
    window = VOICE_BIOMETRIC_PRE_SECONDS + VOICE_BIOMETRIC_POST_SECONDS
    audio = _snapshot_voice_audio(window, sr)
    if audio is None or audio.size == 0:
        return None, 0.0
    _ensure_core_on_path()
    try:
        from core import voice_id
    except Exception as e:
        print(f"  [wake-listener] voice_id import failed: {e}")
        return None, 0.0
    # Snapshot the module-global threshold before taking the lock so a
    # concurrent wake_listener_configure() cannot change it between our
    # write and restore — that would otherwise leave a stale value in
    # voice_id.CONFIDENCE_THRESHOLD for all subsequent callers.
    desired_threshold = float(VOICE_BIOMETRIC_THRESHOLD)
    with _voice_id_lock:
        old_threshold = getattr(voice_id, "CONFIDENCE_THRESHOLD", 0.72)
        try:
            voice_id.CONFIDENCE_THRESHOLD = desired_threshold
            try:
                speaker, score = voice_id.identify_speaker(audio, sr)
            except Exception as e:
                print(f"  [wake-listener] identify_speaker raised: {e}")
                return None, 0.0
        finally:
            voice_id.CONFIDENCE_THRESHOLD = old_threshold
    return speaker, float(score)


def _gate_is_strict() -> bool:
    """True iff the wake event should be rejected when voice ID returns
    no match. False whenever any precondition for strict gating fails —
    that's the permissive fallback path."""
    if not VOICE_BIOMETRIC_ENABLED:
        return False
    if GUEST_MODE_ENABLED:
        return False
    _ensure_core_on_path()
    try:
        from core import voice_id
    except Exception:
        return False
    try:
        if not voice_id.is_available():
            return False
        if not voice_id.list_enrolled():
            return False
    except Exception:
        return False
    return True


def _gate_and_announce(evt: dict) -> None:
    """Background-thread worker — verify the speaker (if gating is
    strict) and, when the main loop is sleeping, nudge it awake via
    proactive_announce. Never blocks the detector callback."""
    phrase = evt.get("phrase", "?")
    if _gate_is_strict():
        speaker, voice_score = _identify_recent_speaker()
        if speaker is None:
            print(f"  [wake-listener] REJECTED '{phrase}' — voice did "
                  f"not match enrolled speaker "
                  f"(voice_score={voice_score:.2f}, "
                  f"threshold={VOICE_BIOMETRIC_THRESHOLD:.2f})")
            return
        print(f"  [wake-listener] verified speaker '{speaker}' "
              f"(voice_score={voice_score:.2f})")

    bobert = sys.modules.get("bobert_companion") or sys.modules.get("__main__")

    # ── Barge-in: wake hit while JARVIS is speaking ──────────────────────
    # request_tts_interrupt() is the monolith's barge-in entry point (see
    # bobert_companion, feat/barge-in). It returns True ONLY when the
    # core.config.BARGE_IN_ENABLED knob is on, TTS playback is actually
    # live, AND the sentence currently being voiced does not itself contain
    # "jarvis" (the echo gate — the mic hears the speakers). On acceptance
    # we return WITHOUT announcing: JARVIS cuts his reply and goes quiet to
    # listen; speaking "Yes, sir?" over the aborted tail would be noise.
    # On False (knob off / not speaking / echo-gated) the legacy wake path
    # below runs completely unchanged. This hook sits AFTER the biometric
    # gate on purpose: an enrolled voiceprint further hardens the echo
    # story, since JARVIS's TTS voice won't match the owner's embedding.
    interrupt = getattr(bobert, "request_tts_interrupt", None)
    if callable(interrupt):
        try:
            if interrupt(source="wake-listener"):
                print(f"  [wake-listener] barge-in on '{phrase}' — "
                      f"TTS cut, staying quiet to listen")
                return
        except Exception as e:
            # Barge-in is best-effort; never let it break the wake path.
            print(f"  [wake-listener] request_tts_interrupt failed: {e}")

    # `_sleep_mode` is a 1-element list ([False]/[True]) in bobert_companion,
    # accessed as _sleep_mode[0] (see self_diagnostic.py ~1041). A non-empty
    # list is always truthy, so the old bool read treated JARVIS as perpetually
    # asleep. Read the first element safely instead.
    sm = getattr(bobert, "_sleep_mode", None)
    asleep = bool(sm[0]) if isinstance(sm, (list, tuple)) and sm else False
    if bobert is not None and asleep:
        announce = getattr(bobert, "proactive_announce", None)
        if callable(announce):
            try:
                announce("Yes, sir?", source="wake-listener")
            except Exception:
                pass


def _on_detect(evt: dict) -> None:
    """Detector callback — log + dispatch the gating/announce work to a
    daemon thread so the PortAudio callback stays non-blocking."""
    phrase = evt.get("phrase", "?")
    score = evt.get("score", 0.0)
    print(f"  [wake-listener] HEARD '{phrase}' "
          f"(score={score:.2f})")
    t = threading.Thread(
        target=_gate_and_announce,
        args=(evt,),
        name="wake-listener-gate",
        daemon=True,
    )
    t.start()


def _get_detector():
    """Lazy import so a missing openwakeword/porcupine install can't
    crash skill loading."""
    global _detector
    if _detector is not None:
        return _detector
    _ensure_core_on_path()
    try:
        from core.wake_word import WakeWordDetector
    except Exception as e:
        print(f"  [wake-listener] core.wake_word import failed: {e}")
        return None
    _detector = WakeWordDetector(
        engine=WAKE_WORD_ENGINE,
        wake_words=WAKE_WORDS,
        threshold=WAKE_WORD_THRESHOLD,
        device=WAKE_WORD_DEVICE,
        on_detect=_on_detect,
        use_silero_vad=WAKE_WORD_USE_SILERO_VAD,
    )
    return _detector


def wake_listener_start(_: str = "") -> str:
    """Start the wake-word detector. Idempotent."""
    with _lock:
        det = _get_detector()
        if det is None:
            return ("Wake-word detector unavailable, sir — "
                    "install openwakeword or pvporcupine.")
        if det.is_running():
            return "Wake-word detector already running, sir."
        ok = det.start()
        if not ok:
            return (f"I'm afraid the {WAKE_WORD_ENGINE} engine failed "
                    "to start, sir. Falling back to hotkey activation.")
        _start_voice_tap(det)
        return f"Listening for {', '.join(WAKE_WORDS)}, sir."


def wake_listener_stop(_: str = "") -> str:
    with _lock:
        global _detector
        if _detector is None or not _detector.is_running():
            return "Wake-word detector is not running, sir."
        _stop_voice_tap()
        _detector.stop()
        # Drop the stopped detector so a later wake_listener_start() rebuilds a
        # fresh one via _get_detector() instead of reusing a stopped object.
        # Mirrors the reconfigure() path, and is why the `global` is here.
        _detector = None
        return "Wake-word detector stopped, sir."


def wake_listener_status(_: str = "") -> str:
    det = _detector
    enrolled_n = 0
    try:
        _ensure_core_on_path()
        from core import voice_id
        enrolled_n = len(voice_id.list_enrolled())
    except Exception:
        pass
    gate = ("guest" if GUEST_MODE_ENABLED
            else ("on" if VOICE_BIOMETRIC_ENABLED else "off"))
    gate_str = (f"gate {gate} (threshold {VOICE_BIOMETRIC_THRESHOLD:.2f}, "
                f"{enrolled_n} enrolled)")
    if det is None:
        return (f"Wake-word detector not initialised — engine "
                f"'{WAKE_WORD_ENGINE}', words {WAKE_WORDS}, {gate_str}.")
    s = det.status()
    state = "active" if s["running"] else "idle"
    last = s["last_event_ts"]
    last_str = (time.strftime("%H:%M:%S", time.localtime(last))
                if last else "never")
    return (f"Wake-word detector {state} — engine {s['engine']}, "
            f"phrases {s['wake_words']}, threshold "
            f"{s['threshold']:.2f}, last hit {last_str}, {gate_str}.")


def wake_listener_configure(arg: str = "") -> str:
    """Adjust runtime config. Argument is 'key=value' — supported keys:
    engine, threshold, words (comma-separated), device, silero (bool),
    voice_gate (bool), voice_threshold (0..1), guest_mode (bool).
    Restart the detector afterwards for engine/words/device changes to
    take effect; gating keys apply immediately."""
    global WAKE_WORD_ENGINE, WAKE_WORD_THRESHOLD, WAKE_WORDS
    global WAKE_WORD_DEVICE, WAKE_WORD_USE_SILERO_VAD, _detector
    global VOICE_BIOMETRIC_ENABLED, VOICE_BIOMETRIC_THRESHOLD
    global GUEST_MODE_ENABLED

    if "=" not in (arg or ""):
        return ("Usage: wake_listener_configure key=value "
                "(engine|threshold|words|device|silero|"
                "voice_gate|voice_threshold|guest_mode).")
    key, _, value = arg.partition("=")
    key = key.strip().lower()
    value = value.strip()

    restart_needed = False
    if key == "engine":
        WAKE_WORD_ENGINE = value or "openwakeword"
        restart_needed = True
    elif key == "threshold":
        try:
            WAKE_WORD_THRESHOLD = max(0.0, min(1.0, float(value)))
        except ValueError:
            return "Threshold must be a number between 0 and 1, sir."
        restart_needed = True
    elif key in {"words", "phrases", "wake_words"}:
        WAKE_WORDS = [w.strip().lower() for w in value.split(",") if w.strip()]
        if not WAKE_WORDS:
            WAKE_WORDS = ["hey jarvis", "jarvis"]
        restart_needed = True
    elif key == "device":
        WAKE_WORD_DEVICE = None if value.lower() in {"", "none", "auto"} else int(value)
        restart_needed = True
    elif key in {"silero", "silero_vad", "vad"}:
        WAKE_WORD_USE_SILERO_VAD = value.lower() in {"1", "true", "yes", "on"}
        restart_needed = True
    elif key in {"voice_gate", "voice_biometric", "biometric"}:
        VOICE_BIOMETRIC_ENABLED = value.lower() in {"1", "true", "yes", "on"}
    elif key in {"voice_threshold", "biometric_threshold"}:
        try:
            new_voice_threshold = max(0.0, min(1.0, float(value)))
        except ValueError:
            return "Voice threshold must be a number between 0 and 1, sir."
        with _voice_id_lock:
            VOICE_BIOMETRIC_THRESHOLD = new_voice_threshold
    elif key in {"guest_mode", "guest"}:
        GUEST_MODE_ENABLED = value.lower() in {"1", "true", "yes", "on"}
    else:
        return f"Unknown wake-listener key '{key}', sir."

    if restart_needed and _detector is not None:
        try:
            _stop_voice_tap()
        except Exception:
            pass
        try:
            _detector.stop()
        except Exception:
            pass
        _detector = None
        return f"Wake-listener {key} set to {value!r}, sir. Restart to apply."
    return f"Wake-listener {key} set to {value!r}, sir."


def guest_mode_on(_: str = "") -> str:
    """Bypass voice gating temporarily for visitors. Resets on restart."""
    global GUEST_MODE_ENABLED
    GUEST_MODE_ENABLED = True
    return ("Guest mode engaged, sir — voice gating is bypassed until "
            "you say 'guest mode off' or the listener restarts.")


def guest_mode_off(_: str = "") -> str:
    """Re-enable voice gating after a guest session."""
    global GUEST_MODE_ENABLED
    GUEST_MODE_ENABLED = False
    return "Guest mode disengaged, sir. Voice gating restored."


def voice_gating_on(_: str = "") -> str:
    global VOICE_BIOMETRIC_ENABLED
    VOICE_BIOMETRIC_ENABLED = True
    return ("Voice biometric gating enabled, sir "
            f"(threshold {VOICE_BIOMETRIC_THRESHOLD:.2f}).")


def voice_gating_off(_: str = "") -> str:
    global VOICE_BIOMETRIC_ENABLED
    VOICE_BIOMETRIC_ENABLED = False
    return ("Voice biometric gating disabled, sir — every wake event "
            "will trigger the recording pipeline.")


def register(actions: dict) -> None:
    """Skill-loader entry point — registers actions and (optionally)
    autostarts the detector. Autostart failures are non-fatal."""
    _apply_bobert_overrides()
    actions["wake_listener_start"] = wake_listener_start
    actions["wake_listener_stop"] = wake_listener_stop
    actions["wake_listener_status"] = wake_listener_status
    actions["wake_listener_configure"] = wake_listener_configure
    actions["guest_mode_on"] = guest_mode_on
    actions["guest_mode_off"] = guest_mode_off
    actions["voice_gating_on"] = voice_gating_on
    actions["voice_gating_off"] = voice_gating_off

    # Don't autostart the wake-word detector when the mic is hard-disabled
    # (staging green candidate / MICROPHONE_INDEX < 0) — it opens its own
    # persistent InputStream (core/wake_word.py) which, on a disabled/None
    # device, would capture the SYSTEM DEFAULT mic and listen.
    _bc = sys.modules.get("bobert_companion") or sys.modules.get("__main__")
    _mic_off = getattr(_bc, "_mic_input_disabled", lambda: False)
    if WAKE_WORD_AUTOSTART and WAKE_WORD_ENGINE != "off" and not _mic_off():
        def _bg():
            try:
                time.sleep(2.0)  # let whisper + skill loader settle
                wake_listener_start("")
            except Exception as e:
                print(f"  [wake-listener] autostart failed: {e}")
        t = threading.Thread(target=_bg, name="wake-listener-autostart",
                             daemon=True)
        t.start()
    else:
        # The detector deliberately does NOT autostart (WAKE_WORD_AUTOSTART
        # above — its InputStream collides with record_speech() on the same
        # WASAPI device), yet _gate_and_announce() here is the ONLY acoustic
        # caller of request_tts_interrupt. So when the owner's barge-in
        # toggle is on, voice barge-in is silently dead until they start the
        # detector by hand. Make the mismatch visible in the session log
        # (2026-07-21 audit). Best-effort: never let it break register().
        try:
            from core import config as _core_cfg
            if getattr(_core_cfg, "BARGE_IN_ENABLED", False):
                print("  [wake-listener] barge-in is enabled but the wake "
                      "detector is not running — say 'start listening for "
                      "the wake word' to arm voice barge-in")
        except Exception:
            pass
