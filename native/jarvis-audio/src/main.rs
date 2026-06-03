//! `jarvis-audio` — the native always-on audio capture/wake/VAD service (M1).
//!
//! STATUS: scaffold. The IPC protocol (`protocol.rs`) is implemented + unit-
//! tested; cpal capture + openWakeWord/Silero VAD + the Windows named-pipe
//! transport land in the next increment (Phase 1 "shadow mode" per the design —
//! run alongside the Python path, logging only). For now this binary parses its
//! args and prints the handshake frame it will emit, so the crate builds + runs
//! end-to-end and the wire contract is exercised from a real entry point.

use jarvis_audio::capture;
use jarvis_audio::protocol::{Event, PROTOCOL_VERSION};

fn pipe_arg() -> String {
    // `--pipe <name>` overrides the default named-pipe path.
    let mut args = std::env::args().skip_while(|a| a != "--pipe");
    args.next(); // consume "--pipe"
    args.next().unwrap_or_else(|| r"\\.\pipe\jarvis-audio".to_string())
}

fn main() {
    let pipe = pipe_arg();
    eprintln!("[jarvis-audio] cpal capture wired; wake/VAD + named-pipe transport next.");
    eprintln!("[jarvis-audio] target pipe: {pipe}");

    match capture::default_input_description() {
        Ok(desc) => eprintln!("[jarvis-audio] default input device: {desc}"),
        Err(e) => eprintln!("[jarvis-audio] no usable input device: {e}"),
    }

    // Emit the handshake the real service leads with, so the protocol path is
    // exercised end-to-end.
    let hello = Event::Hello(PROTOCOL_VERSION).encode();
    eprintln!(
        "[jarvis-audio] Hello frame (v{PROTOCOL_VERSION}, {} bytes): {:02x?}",
        hello.len(),
        hello
    );
}
