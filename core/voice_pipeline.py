"""Voice-pipeline selector — the CI-testable decision layer in front of the two
experimental low-latency voice subsystems.

WHY THIS MODULE EXISTS
----------------------
Two fully-built but unwired subsystems sit beside the ~13K-line monolith:

  * core/realtime_voice.py — a streaming RealtimeSTT→RealtimeTTS pipeline with
    single-queue barge-in (sub-500 ms perceived latency).
  * core/wake_word.py      — a neural wake-word detector (openWakeWord / Porcupine
    + optional Silero VAD).

Both are opt-in and gated behind config flags that DEFAULT OFF (VOICE_MODE and
WAKE_WORD_AUTOSTART in core/config.py). The monolith's hot path must stay
byte-for-byte unchanged when the flags are off, and must fall back to the
existing path on ANY error (a missing optional pip dep, an init failure).

Rather than thread that branching logic — flag read + optional-dep probe +
construct-or-None — through the monolith (where it can't be unit-tested on the
light-deps CI runner), it lives HERE as a handful of small pure-ish functions:

    realtime_enabled() -> bool
    make_realtime_session(**hooks) -> object | None
    wake_word_autostart_enabled() -> bool
    wake_detector_or_none(**opts) -> object | None
    deps_status() -> dict           # diagnostics: which optional deps resolve

The monolith just calls these and branches, wrapping the call in a try/except
that returns to the existing path on failure. The SELECTION logic is therefore
exercised by tests/test_voice_pipeline.py on CI (stdlib-only, deps absent →
returns None/False + logs), while the monolith diff stays a thin, obvious call.

CI-SAFETY CONTRACT
------------------
* Importing this module is ALWAYS safe: top level imports only stdlib +
  core.config. The heavy/optional modules (core.realtime_voice, core.wake_word)
  and their pip deps (RealtimeSTT/RealtimeTTS/openwakeword/pvporcupine) are
  imported lazily INSIDE the functions, each wrapped so a missing dep yields
  None + a single log line, never an exception.
* Optional deps are probed via importlib.util.find_spec (never a bare unused
  import), so this file is pyflakes-clean on a bare runner.
* Every public function is total: it returns a value for every input and never
  propagates an exception to its caller.

ENV OVERRIDES
-------------
Each flag honours an env var that WINS over the core.config constant, matching
the rest of JARVIS's "constant is the default, JARVIS_* env overrides it"
convention:

    JARVIS_VOICE_MODE           overrides VOICE_MODE          (e.g. "realtime")
    JARVIS_WAKE_WORD_AUTOSTART  overrides WAKE_WORD_AUTOSTART  (truthy → on)
"""

from __future__ import annotations

import importlib.util
import os
from typing import Callable, Optional

from core import config


# Optional pip deps each subsystem needs. Used ONLY for find_spec probes in
# deps_status() / the *_available() helpers — never imported here.
_REALTIME_DEPS = ("RealtimeSTT", "RealtimeTTS")
_WAKE_WORD_DEPS = ("openwakeword", "pvporcupine")

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f", ""}


def _log(msg: str) -> None:
    """Single channel for the module's diagnostics so the monolith's stdout
    stays consistent with the rest of the voice subsystem ('  [voice-pipeline]
    ...'). Kept tiny + import-free so callers can monkeypatch builtins.print in
    tests without side effects."""
    print(f"  [voice-pipeline] {msg}")


def _cfg(name: str, default):
    """Read a config knob, letting a matching JARVIS_<NAME> env var override it.

    The env value is only consulted when present AND non-empty; otherwise the
    core.config constant (or the supplied default if the constant is absent on
    an older/newer tree) is used. This mirrors the project-wide override
    convention without importing anything heavy."""
    env = os.environ.get("JARVIS_" + name)
    if env is not None and env.strip() != "":
        return env.strip()
    return getattr(config, name, default)


def _as_bool(value, default: bool = False) -> bool:
    """Coerce a config/env value to bool. Real bools pass through; strings are
    matched case-insensitively against the truthy/falsy sets; anything else
    falls back to `default`. Never raises."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUTHY:
            return True
        if v in _FALSY:
            return False
    return default


def _spec_present(name: str) -> bool:
    """True iff importlib can locate `name` WITHOUT importing it. Tolerates the
    ValueError/ModuleNotFoundError that find_spec can raise for half-initialised
    or namespace-shadowed packages — treated as 'absent'."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _all_specs_present(names) -> bool:
    return all(_spec_present(n) for n in names)


# ──────────────────────────────────────────────────────────────────────
#  F1 — realtime streaming voice (behind VOICE_MODE == "realtime")
# ──────────────────────────────────────────────────────────────────────

def realtime_enabled() -> bool:
    """True iff VOICE_MODE (or the JARVIS_VOICE_MODE override) selects the
    realtime streaming pipeline. Pure flag check — does NOT probe deps or touch
    audio, so it's cheap to call in the boot path. Default 'turn_based' → False.
    """
    mode = _cfg("VOICE_MODE", "turn_based")
    return str(mode).strip().lower() == "realtime"


def realtime_available() -> tuple[bool, str]:
    """Return (True, '') iff both realtime pip deps resolve, else (False,
    reason). Probes via find_spec only — importing nothing — so it's safe and
    fast on a bare runner where the deps are absent."""
    missing = [n for n in _REALTIME_DEPS if not _spec_present(n)]
    if missing:
        return False, "missing optional dep(s): " + ", ".join(missing)
    return True, ""


def make_realtime_session(
    *,
    on_user_utterance: Optional[Callable[[str], None]] = None,
    on_partial_transcript: Optional[Callable[[str], None]] = None,
    on_barge_in: Optional[Callable[[], None]] = None,
    stt_model: Optional[str] = None,
    stt_language: str = "en",
    tts_engine: str = "system",
    tts_voice: Optional[str] = None,
    input_device: Optional[int] = None,
) -> Optional[object]:
    """Build + START a realtime voice session, or return None to signal 'stay on
    the turn-based loop'.

    Returns None (and logs exactly one line) when ANY of these hold:
      * the flag is off (VOICE_MODE != 'realtime'),
      * the optional deps are absent,
      * core.realtime_voice can't be imported,
      * the pipeline fails to start (no mic, bad engine, init error).

    On success returns the live RealtimeVoicePipeline (already started, its STT
    thread up). The caller branches into the streaming path ONLY on a truthy
    return and otherwise behaves exactly as the historical loop — so a missing
    dep or a cold-start failure transparently degrades to turn_based.

    Total: never raises. Any unexpected error is caught, logged, and rendered as
    None so the monolith's fallback path engages.
    """
    if not realtime_enabled():
        # Off by default — silent, no dep probing. This is the hot path.
        return None

    ok, why = realtime_available()
    if not ok:
        _log(f"VOICE_MODE=realtime but {why}; falling back to turn_based")
        return None

    try:
        from core import realtime_voice as rtv
    except Exception as e:  # pragma: no cover - import guarded for safety
        _log(f"core.realtime_voice import failed ({e}); falling back to turn_based")
        return None

    # Resolve the few tunables that default to the realtime module's own
    # constants / the live TTS voice when the caller doesn't override them.
    model = stt_model or getattr(rtv, "DEFAULT_STT_MODEL", "base")
    voice = tts_voice or getattr(config, "TTS_VOICE", None) \
        or getattr(rtv, "DEFAULT_TTS_VOICE", "en-GB-RyanNeural")

    try:
        # start_pipeline honours voice_mode internally AND returns None on a
        # start() failure, so it already encodes most of our fallback contract;
        # we pass voice_mode='realtime' explicitly because we've decided to
        # engage. A truthy return is a running pipeline.
        pipe = rtv.start_pipeline(
            voice_mode="realtime",
            on_user_utterance=on_user_utterance,
            on_partial_transcript=on_partial_transcript,
            on_barge_in=on_barge_in,
            stt_model=model,
            stt_language=stt_language,
            tts_engine=tts_engine,
            tts_voice=voice,
            input_device=input_device,
        )
    except Exception as e:
        _log(f"realtime pipeline start raised ({e}); falling back to turn_based")
        return None

    if pipe is None:
        _log("realtime pipeline unavailable at start; falling back to turn_based")
        return None
    _log("realtime streaming voice engaged")
    return pipe


# ──────────────────────────────────────────────────────────────────────
#  F2 — neural wake detector in standby (behind WAKE_WORD_AUTOSTART)
# ──────────────────────────────────────────────────────────────────────

def wake_word_autostart_enabled() -> bool:
    """True iff WAKE_WORD_AUTOSTART (or the JARVIS_WAKE_WORD_AUTOSTART override)
    is on. Pure flag check; default False. Gates the neural-detector standby
    path in _handle_sleep_standby."""
    return _as_bool(_cfg("WAKE_WORD_AUTOSTART", False), default=False)


def wake_word_engine() -> str:
    """The wake-word engine to instantiate ('openwakeword' | 'porcupine' |
    'off'). Defaults to 'openwakeword' but honours a JARVIS_WAKE_WORD_ENGINE
    override so a user with a Porcupine key can switch without editing config.
    """
    return str(_cfg("WAKE_WORD_ENGINE", "openwakeword")).strip().lower()


def wake_word_available(engine: Optional[str] = None) -> tuple[bool, str]:
    """Return (True, '') iff the chosen engine's pip dep resolves, else (False,
    reason). 'openwakeword' needs the openwakeword package; 'porcupine' needs
    pvporcupine. find_spec-only, imports nothing."""
    eng = (engine or wake_word_engine()).strip().lower()
    if eng == "off":
        return False, "engine=off"
    if eng == "openwakeword":
        return (True, "") if _spec_present("openwakeword") \
            else (False, "openwakeword not installed")
    if eng == "porcupine":
        return (True, "") if _spec_present("pvporcupine") \
            else (False, "pvporcupine not installed")
    return False, f"unknown engine '{eng}'"


def make_wake_detector(
    *,
    on_detect: Optional[Callable[[dict], None]] = None,
    wake_words: Optional[list] = None,
    threshold: Optional[float] = None,
    device: Optional[int] = None,
    sample_rate: int = 16000,
    engine: Optional[str] = None,
    use_silero_vad: bool = False,
    autostart: bool = True,
) -> Optional[object]:
    """Construct (and, by default, START) a WakeWordDetector — or return None to
    signal 'use the existing Whisper-substring standby path'.

    Returns None (and logs one line) when ANY of these hold:
      * the autostart flag is off (WAKE_WORD_AUTOSTART False),
      * the engine is 'off' or its optional dep is absent,
      * core.wake_word can't be imported,
      * the detector fails to construct or (when autostart=True) fails to start.

    On success returns the detector. With autostart=True the returned detector
    is already running its background InputStream; pass autostart=False to get a
    constructed-but-idle detector (the monolith uses this to feed the detector
    frames it already captured, rather than opening a second mic stream).

    Total: never raises; unexpected errors degrade to None so the caller keeps
    the Whisper path.
    """
    if not wake_word_autostart_enabled():
        # Off by default — silent, no dep probing. Hot path for standby.
        return None

    eng = (engine or wake_word_engine()).strip().lower()
    ok, why = wake_word_available(eng)
    if not ok:
        _log(f"WAKE_WORD_AUTOSTART on but {why}; keeping Whisper standby path")
        return None

    try:
        from core.wake_word import WakeWordDetector
    except Exception as e:  # pragma: no cover - import guarded for safety
        _log(f"core.wake_word import failed ({e}); keeping Whisper standby path")
        return None

    try:
        kwargs = dict(engine=eng, sample_rate=sample_rate,
                      on_detect=on_detect, use_silero_vad=use_silero_vad)
        if wake_words is not None:
            kwargs["wake_words"] = list(wake_words)
        if threshold is not None:
            kwargs["threshold"] = float(threshold)
        if device is not None:
            kwargs["device"] = device
        detector = WakeWordDetector(**kwargs)
    except Exception as e:
        _log(f"wake detector construct failed ({e}); keeping Whisper standby path")
        return None

    if not autostart:
        return detector

    try:
        started = detector.start()
    except Exception as e:
        _log(f"wake detector start raised ({e}); keeping Whisper standby path")
        return None
    if not started:
        _log("wake detector failed to start; keeping Whisper standby path")
        return None
    _log(f"neural wake detector engaged (engine={eng})")
    return detector


# Back-compat / spec-named alias: the architecture note calls for a
# `wake_detector_or_none()` accessor. Keep it as a thin wrapper so callers can
# use either name.
def wake_detector_or_none(**opts) -> Optional[object]:
    """Alias for make_wake_detector(**opts) — returns the detector or None."""
    return make_wake_detector(**opts)


# ──────────────────────────────────────────────────────────────────────
#  Diagnostics
# ──────────────────────────────────────────────────────────────────────

def deps_status() -> dict:
    """A flag+dep snapshot for the self-diagnostic / a 'voice status' action.
    Pure: only reads flags and runs find_spec probes — opens nothing, imports
    nothing heavy."""
    rt_ok, rt_why = realtime_available()
    ww_ok, ww_why = wake_word_available()
    return {
        "voice_mode": str(_cfg("VOICE_MODE", "turn_based")),
        "realtime_enabled": realtime_enabled(),
        "realtime_available": rt_ok,
        "realtime_detail": rt_why,
        "realtime_deps": {n: _spec_present(n) for n in _REALTIME_DEPS},
        "wake_word_autostart": wake_word_autostart_enabled(),
        "wake_word_engine": wake_word_engine(),
        "wake_word_available": ww_ok,
        "wake_word_detail": ww_why,
        "wake_word_deps": {n: _spec_present(n) for n in _WAKE_WORD_DEPS},
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import json
    print("voice_pipeline selector smoke test")
    print(json.dumps(deps_status(), indent=2))
    print(f"  realtime_enabled()           = {realtime_enabled()}")
    print(f"  make_realtime_session()      = {make_realtime_session()!r}")
    print(f"  wake_word_autostart_enabled()= {wake_word_autostart_enabled()}")
    print(f"  wake_detector_or_none()      = {wake_detector_or_none()!r}")
