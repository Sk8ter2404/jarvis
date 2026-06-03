//! JARVIS native audio service library.
//!
//! The crate is the roadmap **M1** "native always-on audio + wake + VAD service"
//! (see `docs/design/M1-native-audio-service.md`). This first cut implements the
//! **IPC protocol** — the wire contract between this service and the Python
//! "brain" — and is fully unit-tested. The capture/wake/VAD pipeline (cpal +
//! openWakeWord/Silero) and the Windows named-pipe transport land in the next
//! increment, behind the same additive/shadow-mode plan the design lays out
//! (the Python capture path stays the live fallback the whole time).
pub mod protocol;
