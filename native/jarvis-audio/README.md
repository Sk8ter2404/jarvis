# jarvis-audio — native audio service (roadmap M1)

The native, always-on **audio capture + wake-word + VAD** service for JARVIS.
It runs as a child process of the Python "brain", owns the microphone, and hands
Python only *post-wake, endpointed PCM* over a local named pipe — getting the
soft-real-time audio path off the Python GIL.

Design + rationale (incl. the Rust-vs-Go decision): see
[`../../docs/design/M1-native-audio-service.md`](../../docs/design/M1-native-audio-service.md).

## Status

**Scaffold.** This first cut implements + unit-tests the **IPC protocol**
(`src/protocol.rs`) — the wire contract between the service and Python — and a
buildable service skeleton (`src/main.rs`). Still to come, additively (the
Python capture path stays the live fallback the whole time):

1. cpal ring-buffer capture (WASAPI on Windows).
2. openWakeWord / Porcupine wake detection + Silero/WebRTC VAD + endpointing.
3. The Windows named-pipe transport (`\\.\pipe\jarvis-audio`) + a Python client.
4. Shadow mode → opt-in cutover behind `AUDIO_SERVICE_ENABLED` (default off).

## Build & test

Needs the Rust toolchain (`rustup`, stable). The Python CI does **not** build
this crate — it has its own cargo build/test.

```powershell
cd native/jarvis-audio
cargo test     # protocol round-trip + framing tests
cargo build    # debug binary at target/debug/jarvis-audio.exe
cargo run -- --pipe \\.\pipe\jarvis-audio
```
