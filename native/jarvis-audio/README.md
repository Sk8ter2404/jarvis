# jarvis-audio — native audio service (roadmap M1)

The native, always-on **audio capture + wake-word + VAD** service for JARVIS.
It runs as a child process of the Python "brain", owns the microphone, and hands
Python only *post-wake, endpointed PCM* over a local named pipe — getting the
soft-real-time audio path off the Python GIL.

Design + rationale (incl. the Rust-vs-Go decision): see
[`../../docs/design/M1-native-audio-service.md`](../../docs/design/M1-native-audio-service.md).

## Status

**Early.** Implemented + unit-tested so far: the **IPC protocol**
(`src/protocol.rs`, the Rust↔Python wire contract), an audio **ring buffer**
(`src/ring_buffer.rs`), and **cpal capture** (`src/capture.rs` — opens the
default input device and converts its samples to 16-bit PCM into the ring
buffer; `cargo run` prints the detected device). Still to come, additively (the
Python capture path stays the live fallback the whole time):

1. openWakeWord / Porcupine **neural wake** detection (replacing the energy gate
   in `src/vad.rs`, which already does RMS VAD + endpointing) + an optional
   Silero neural VAD.
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
