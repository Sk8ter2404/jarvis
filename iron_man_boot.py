#!/usr/bin/env python3
"""
Iron Man-style JARVIS boot sequence.

Spec (jarvis_todo.md 2026-05-27 10:04, iron_man_boot): replace the previous
silent skill-registration startup with something that feels like the suit
powering on. Three coordinated pieces:

  1. A ~1.5s suit power-on audio sting (synthesised on the fly so we don't
     ship a copyrighted sample). Plays through the configured output device.
  2. A brief HUD overlay animation on the top monitor — the arc-reactor
     expands while the center label scrolls
         INITIALISING  →  DIAGNOSTICS  →  ONLINE
     (the HUD itself drives the visual; this module just sets boot_phase /
     boot_started_at / boot_duration in hud_state.json so the HUD takes
     over its canvas for the duration).
  3. A single TTS line —
         "JARVIS online. All systems nominal. Good <time-of-day>, sir."
     spoken once the sting has cleared the speakers.

Returns the spoken line so the caller can append it to conversation_history.

This module is intentionally self-contained: callers pass in `speak_fn` and
the HUD-state writer, so the file doesn't import anything from
bobert_companion. The audio sting falls back gracefully if sounddevice or
numpy aren't importable — only the HUD animation + spoken line will run.
"""
import datetime as _dt
import math
import os
import time

# Animation length (seconds) — the HUD shows the rings filling for this long
# and self-clears 0.5s after expiry so a crashed parent can't strand the
# overlay. Tuned to match a typical 3-clause TTS line so the rings finish
# expanding right around when JARVIS stops speaking.
BOOT_ANIMATION_SECONDS = 4.5

# Sting duration — the spec asks for ~1.5s. Kept short so we don't talk
# over the start of the TTS line.
STING_DURATION_SECONDS = 1.5

# After speech finishes, ensure the animation has been visible at least this
# long so the rings get a chance to fill all the way out even on a fast
# TTS backend.
MIN_VISIBLE_SECONDS = 4.0

# Where an override sting WAV can live. If this file is present the user's
# own sample is used in place of the synthesised one. The synth fallback
# always exists so a stock checkout still gets the full effect.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STING_PATH = os.path.join(_PROJECT_DIR, "data", "iron_man_boot.wav")

# Hold a strong reference to the most recently played sting buffer at module
# scope so PortAudio's background mixer thread can keep reading it even after
# play_iron_man_boot's locals have been released. sd.play() is non-blocking
# and the numpy buffer must outlive the call by the full sting duration —
# without this anchor, fast TTS paths can return before the sting decays and
# the buffer becomes GC-eligible while PortAudio is still copying from it
# (0xc0000374 heap corruption on Windows).
_LAST_STING_BUF = None


def _time_of_day_phrase(now: _dt.datetime | None = None) -> str:
    """Map the current local hour to one of morning/afternoon/evening.
    Late night (22:00–04:59) collapses into 'evening' rather than introducing
    a fourth bucket — the spoken line should never say 'good night' at
    midnight while the user is still working."""
    h = (now or _dt.datetime.now()).hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    return "evening"


def build_boot_line(now: _dt.datetime | None = None) -> str:
    """Compose the single spoken line. Exposed so tests can pin a phrase."""
    return f"JARVIS online. All systems nominal. Good {_time_of_day_phrase(now)}, sir."


def _synthesise_sting(duration: float = STING_DURATION_SECONDS,
                      sample_rate: int = 44100):
    """Generate a 'suit power-on' sting as a float32 mono numpy array.

    Designed in three layers so it reads as a confident mechanical power-up:
      • body — a sine sweep 90 Hz → 480 Hz with second + third harmonics,
        envelope ramping in fast and tailing off (the rising whine).
      • motor — a slow amplitude modulation at ~6 Hz on the body (suggests
        a spinning servo coming up to speed).
      • bloom — a short bright chime at 0.95× duration (E5 + B5 sines, fast
        attack, exponential decay) that lands just before the spoken line.

    Returns (audio_float32, sample_rate). Caller is expected to clip / scale
    if the output device needs int16; sounddevice handles float32 directly.
    """
    import numpy as np  # local import so this module imports cleanly on a
                        # numpy-less environment (HUD + speak still work)

    n = int(duration * sample_rate)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)

    # ── body: exponential frequency sweep with light harmonics ──────────
    f_start, f_end = 90.0, 480.0
    # Exponential rather than linear sweep — sounds more "spinning up".
    freq = f_start * (f_end / f_start) ** (t / duration)
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    body = (
        0.55 * np.sin(phase).astype(np.float32)
        + 0.22 * np.sin(2.0 * phase).astype(np.float32)
        + 0.10 * np.sin(3.0 * phase).astype(np.float32)
    )

    # ── motor: 6 Hz tremolo so it feels mechanical, not synth-clean ─────
    motor = 0.85 + 0.15 * np.sin(2.0 * np.pi * 6.0 * t, dtype=np.float32)
    body = body * motor

    # ── body envelope: 25 ms attack, plateau, slow decay ────────────────
    env = np.ones_like(t)
    attack = max(1, int(0.025 * sample_rate))
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    # Decay across the back two-thirds — gentle so the bloom can still ring.
    decay_start = int(0.35 * n)
    decay_len = n - decay_start
    if decay_len > 0:
        env[decay_start:] = np.linspace(1.0, 0.55, decay_len, dtype=np.float32)
    body = body * env

    # ── bloom: bright chime that lands just before the spoken line ──────
    bloom_start = int(0.65 * n)
    bloom_len = n - bloom_start
    if bloom_len > 0:
        tb = np.arange(bloom_len, dtype=np.float32) / sample_rate
        chime = (
            0.30 * np.sin(2.0 * np.pi * 659.25 * tb).astype(np.float32)  # E5
            + 0.18 * np.sin(2.0 * np.pi * 987.77 * tb).astype(np.float32)  # B5
        )
        # Sharp attack, exponential decay over ~250 ms.
        decay = np.exp(-tb * 4.0).astype(np.float32)
        chime = chime * decay
        body[bloom_start:] = body[bloom_start:] + chime

    # ── global level — peak-normalise to 0.65 so we never clip ──────────
    peak = float(np.max(np.abs(body)) or 1.0)
    body = (body * (0.65 / peak)).astype(np.float32)
    return body, sample_rate


def _play_sting_async(audio, sample_rate: int,
                      output_device: int | None = None) -> bool:
    """Kick off non-blocking playback of the sting. Returns True if playback
    started (the caller can keep going while it rings out), False otherwise.

    Uses sounddevice's play() which is already a project dependency and
    drives the same output device the TTS path uses, so the sting and the
    spoken line come out the same speakers."""
    global _LAST_STING_BUF
    try:
        import sounddevice as sd
        kwargs = {"samplerate": sample_rate}
        if output_device is not None:
            kwargs["device"] = output_device
        # Anchor the buffer at module scope BEFORE sd.play so the underlying
        # numpy array can't be reclaimed mid-playback if the caller returns.
        _LAST_STING_BUF = audio
        sd.play(audio, **kwargs)
        return True
    except Exception as e:
        print(f"  [iron_man_boot] sting playback failed: {e}")
        return False


def _load_sting_from_disk(path: str):
    """Best-effort load of a WAV sting from disk. Returns (audio, sr) or
    None on any failure — the caller falls back to the synthesised sting."""
    if not path or not os.path.exists(path):
        return None
    try:
        import soundfile as sf  # already a project dep (TTS pipeline)
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            # Collapse stereo → mono so it matches the synth path.
            audio = audio.mean(axis=1).astype("float32")
        return audio, int(sr)
    except Exception as e:
        print(f"  [iron_man_boot] disk sting load failed ({path}): {e}")
        return None


def play_iron_man_boot(speak_fn,
                       write_hud_state=None,
                       output_device: int | None = None,
                       sting_path: str | None = None,
                       now: _dt.datetime | None = None,
                       tts_muted: bool = False) -> str:
    """Run the full Iron Man boot sequence.

    Args:
        speak_fn: callable(str) → None. Synchronous (blocks until TTS done).
        write_hud_state: optional callable(**fields) → None. Used to publish
            boot_phase / boot_started_at / boot_duration so the HUD can
            render the power-up animation. None disables the visual; the
            audio + spoken line still fire.
        output_device: optional sounddevice device index for the sting.
            None → system default (matches what TTS picks).
        sting_path: optional path to a WAV file to use instead of the
            synthesised sting. Defaults to DEFAULT_STING_PATH; falls back
            to the synth if the file is missing or unreadable.
        now: optional datetime for the time-of-day greeting (testing hook).

    Returns:
        The spoken boot line, so the caller can append it to
        conversation_history.
    """
    boot_line = build_boot_line(now)

    # ── 1. Prepare audio (disk override or synth) ──────────────────────
    # When TTS is muted we skip both the load/synth AND the playback below so
    # the audio device is never opened for the sting — matches _speak()'s
    # short-circuit on _tts_muted (bobert_companion.py:13451).
    if tts_muted:
        sting = None
    else:
        sting = _load_sting_from_disk(sting_path or DEFAULT_STING_PATH)
        if sting is None:
            try:
                sting = _synthesise_sting()
            except Exception as e:
                # numpy/sounddevice missing or broken — sting becomes a no-op.
                print(f"  [iron_man_boot] sting synth failed: {e}")
                sting = None

    # ── 2. Kick off the HUD animation BEFORE the sting so the rings start
    #       expanding at the same instant as the audio. The HUD draws the
    #       INITIALISING / DIAGNOSTICS / ONLINE labels in sync with progress.
    started_at = time.time()
    if write_hud_state is not None:
        try:
            write_hud_state(
                boot_phase="powering",
                boot_started_at=started_at,
                boot_duration=BOOT_ANIMATION_SECONDS,
                state="Initialising",
            )
        except Exception as e:
            print(f"  [iron_man_boot] HUD start publish failed: {e}")

    # ── 3. Play the sting non-blocking so the spoken line can follow on
    #       its tail rather than waiting for it to fully decay.
    if sting is not None:
        audio, sr = sting
        _play_sting_async(audio, sr, output_device=output_device)
        # Wait the FULL sting duration before letting TTS start. The previous
        # partial-wait (0.55× duration) had the sting and TTS overlap for the
        # cinematic effect, but it also let play_iron_man_boot proceed (and
        # speak_fn potentially open its own PortAudio stream) while sd.play
        # was still draining the sting buffer — that overlap was the proximate
        # cause of the boot-time 0xc0000374 crash. The strong reference held
        # in _LAST_STING_BUF + this full wait together guarantee PortAudio is
        # done with the buffer before any other audio path touches the device.
        try:
            time.sleep(STING_DURATION_SECONDS)
        except Exception:
            pass

    # ── 4. Speak the single boot line. _speak() handles set_state /
    #       lip-sync / pending-speech queueing inside the host process.
    try:
        speak_fn(boot_line)
    except Exception as e:
        print(f"  [iron_man_boot] boot line speak failed: {e}")

    # ── 5. Hold the HUD overlay for at least MIN_VISIBLE_SECONDS so the
    #       rings get a chance to fill all the way out even when the TTS
    #       line finishes fast. Then clear the boot_phase so the HUD goes
    #       back to its normal status rendering.
    if write_hud_state is not None:
        try:
            elapsed = time.time() - started_at
            if elapsed < MIN_VISIBLE_SECONDS:
                time.sleep(MIN_VISIBLE_SECONDS - elapsed)
            write_hud_state(
                boot_phase="",
                boot_started_at=0.0,
                state="Idle",
            )
        except Exception as e:
            print(f"  [iron_man_boot] HUD clear failed: {e}")

    return boot_line
