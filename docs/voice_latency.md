# Low-latency voice (experimental, opt-in)

Two already-built subsystems can cut perceived voice latency. **Both default OFF**
— with the flags unset, JARVIS behaves exactly as before (full Whisper turn-based
capture). Each new path falls back to the existing one on any error (missing dep,
init failure), so enabling them can't brick voice — worst case you're back to the
default behavior with one log line.

See [docs/design/M1-native-audio-service.md](design/M1-native-audio-service.md)
for the larger native-audio plan these are the down payment on.

## F1 — streaming voice (`core/realtime_voice.py`)

Replaces the mic→Whisper *input* side with a streaming STT pipeline
(sub-500ms, single-queue barge-in). Reply audio still uses the existing `_speak()`
path (fully streaming TTS output is a later, larger change).

```
pip install RealtimeSTT RealtimeTTS
# then either:
setx JARVIS_VOICE_MODE realtime      # env override, or
# set VOICE_MODE = "realtime" in core/config.py
```

If the deps aren't installed, `core.voice_pipeline.make_realtime_session()`
returns `None` and JARVIS stays on turn-based for the session.

## F2 — neural wake word in standby (`core/wake_word.py`)

In standby, the default path runs a **full GPU Whisper transcription per
utterance** just to match "jarvis". The neural detector does it for ~0 GPU.

```
pip install openwakeword          # default engine
#   or: pip install pvporcupine   # + setx PORCUPINE_ACCESS_KEY <key>
# optional endpointing: pip install silero-vad torch

setx JARVIS_WAKE_WORD_AUTOSTART 1   # env override, or
# set WAKE_WORD_AUTOSTART = True in core/config.py
# engine: JARVIS_WAKE_WORD_ENGINE = openwakeword | porcupine
```

On any detector error the standby loop falls back to the Whisper-substring path.

## How to verify it's safe

`core/voice_pipeline.py` is the selector seam — pure flag checks + `find_spec`
dep probes, every function total (never raises). With both flags unset:
`realtime_enabled()` and `wake_word_autostart_enabled()` return `False`,
`make_realtime_session()` / `make_wake_detector()` return `None`, and the monolith
branch points short-circuit on their first line into the existing code. Covered by
`tests/test_voice_pipeline.py` (47) + `tests/monolith/test_monolith_voice_wiring.py`
(24).
