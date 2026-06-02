# Design: M1 — Native always-on audio + wake + VAD service

Status: **proposal** · Roadmap item: `1.0.0` / M1 · Author: design pass 2026-06-02

> The single largest *felt* latency lever, and the one architectural change that
> needs a language decision. This doc makes the case, recommends a language, and
> phases the migration so it never destabilizes the live daily-driver.

---

## 1. Problem

From the performance survey of the live tree:

1. **The wake path runs a full GPU Whisper inference per standby utterance** just
   to substring-match "jarvis". `_handle_sleep_standby` (`bobert_companion.py`)
   calls `record_speech()` → full `large-v3-turbo` transcription → Python
   `any(wp in tl for wp in WAKE_PHRASES)`. A real neural detector
   (`core/wake_word.py`) exists but is never imported (`WAKE_WORD_AUTOSTART=False`).
2. **GIL contention on the soft-real-time audio path.** Audio capture + the
   per-chunk AEC/NS/AGC DSP, the 20 FPS OpenCV face loop, and any local
   inference all execute Python bytecode under one GIL. The codebase carries scar
   tissue: a `record_speech` hang watchdog, a documented ~70 s mic-stall, and
   `sounddevice.close()` SIGSEGV workarounds.
3. **A fixed 1.4 s silence endpoint** (`SILENCE_SECS`) is added to *every*
   utterance before STT begins.

(The other half of the 3–5 s latency is the two serial un-streamed network legs —
Claude + edge-TTS. That's addressed by **F1** (streaming voice) and is *not* a
language problem; this doc is specifically about the capture/wake/VAD hot path.)

## 2. Goals / non-goals

**Goals**
- Replace "Whisper-as-wake-word" with a tight always-on **wake + VAD** loop that
  costs ~0 GPU and < a few % of one CPU core.
- Get audio capture + endpointing **off the Python GIL** so the face loop, DSP,
  and inference stop contending with capture (kills the jitter/glitch class).
- Hand Python only **post-wake, endpointed PCM** — the brain wakes up already
  holding a clean utterance.
- Sub-300 ms wake-to-capture; barge-in-capable.

**Non-goals (v1)**
- Moving STT, the LLM, skills, memory, or TTS off Python. faster-whisper
  (ctranslate2) already runs in C++/CUDA and releases the GIL — the survey
  confirmed STT is **not** the bottleneck. Keep it Python.
- Replacing the conversation loop or `ACTIONS` dispatch. The brain stays Python.

## 3. Proposed architecture

```
┌────────────────────────────┐         ┌──────────────────────────────┐
│  native audio service      │  IPC    │  Python "brain" (existing)   │
│  (Rust, new)               │ ──────► │  bobert_companion.py loop    │
│                            │ events  │                              │
│  • cpal ring-buffer capture│ + PCM   │  • transcribe()  (faster-    │
│  • openWakeWord/Porcupine  │         │    whisper, unchanged)       │
│  • Silero/WebRTC VAD       │ ◄────── │  • LLM dispatch / ACTIONS    │
│  • endpointing             │ control │  • streaming TTS (F1)        │
│  • AEC tap (later)         │         │  • barge-in signal           │
└────────────────────────────┘         └──────────────────────────────┘
```

- The native service owns the microphone, runs continuously, and emits a small
  stream of **events** + **PCM** to Python over a local IPC channel:
  `wake_detected`, `vad_start`, `audio_frame(pcm)`, `vad_end(utterance)`,
  `barge_in`. Python sends back `mute` / `duck` / `start_listening` /
  `stop` control messages (e.g. mute capture while TTS plays).
- Python's loop replaces the `record_speech()` / `_handle_sleep_standby` blocking
  capture with "await the next `vad_end` utterance from the service", then runs
  the **unchanged** `transcribe()` → LLM → TTS path.
- **The existing Python capture path stays as the fallback** the entire time
  (behind the same flag F2 introduces), so the service is purely additive until
  it's proven and defaulted on.

### IPC

A **local socket** — a Windows **named pipe** (`\\.\pipe\jarvis-audio`), Unix
domain socket elsewhere — carrying a tiny length-prefixed binary protocol:
a 1-byte event tag + payload (control events are a few bytes; `audio_frame` is
16 kHz/16-bit mono PCM). Rationale over the alternatives:
- vs. the existing **JSON-file IPC**: far too slow/racy for streaming audio.
- vs. **gRPC**: overkill; adds a heavy dep + codegen for a 5-message protocol.
- vs. **shared memory ring**: lowest latency but most complex/unsafe; revisit
  only if socket copy cost ever shows up (it won't at 16 kHz mono = 32 KB/s).
The service runs as a **child process** of the brain (like the HUD/tray today),
so lifecycle + the singleton model are unchanged; it dies with JARVIS.

## 4. Language: **Rust** (recommended), with the Go trade-off stated

This is the decision the roadmap flagged. Recommendation: **Rust.**

| Factor | Rust | Go |
|---|---|---|
| **Real-time audio determinism** | **No GC** — no pause can drop a capture frame. The decisive factor: audio glitches/jitter are the *exact* failure mode the survey found. | GC pauses are small (sub-ms) but non-zero; a tight capture+VAD loop is precisely where they bite. |
| Audio I/O | `cpal` — mature, cross-platform (WASAPI/CoreAudio/ALSA). | `portaudio`/`malgo` via cgo — workable, less idiomatic. |
| Wake/VAD models | `ort` (ONNX Runtime) for openWakeWord + Silero; Porcupine has a Rust SDK. | `onnxruntime_go` via cgo; Porcupine Go SDK exists. |
| Footprint / deploy | Single small static binary, tiny RAM. | Single binary, larger runtime, GC heap. |
| Iteration speed | Slower (borrow checker, build times). | **Faster to write**, gentler learning curve. |
| Fit for *this* component | A small, stable, long-lived service that rarely changes once correct — pays Rust's up-front cost once and reaps determinism forever. | Better if the priority were dev velocity / team familiarity. |

**Why Rust wins here specifically:** this is a soft-real-time DSP loop, not a CRUD
service. The whole point of moving it out of Python is determinism on the audio
path; choosing a GC'd runtime would re-introduce a (smaller) version of the very
jitter we're eliminating. Go's GC is good enough that it would *probably* be fine
— so if the owner values iteration speed over the last slice of determinism, Go
is a defensible second choice. But for "build it once, never glitch," **Rust**.

## 5. Migration plan (additive, reversible, never breaks the daily-driver)

- **Phase 0 — down payment (PR #3, in flight):** flag-gated **F1** (streaming
  voice) + **F2** (neural wake *in Python*). Proves the UX win and the flag
  surface with zero native code. If F2-in-Python already removes the
  Whisper-as-wake cost acceptably, M1 can even be deprioritized.
- **Phase 1 — the service, shadow mode:** build the Rust service (capture + wake
  + VAD + endpointing). Run it alongside the Python path, **logging only** —
  compare its wake/endpoint decisions against the live path on real audio for a
  few days. No behavior change.
- **Phase 2 — opt-in cutover:** behind `AUDIO_SERVICE_ENABLED` (default False),
  Python consumes the service's utterances instead of `record_speech()`; the
  Python path remains the instant fallback on any service fault.
- **Phase 3 — default on**, Python capture path retained as fallback for one
  release, then removed.

## 6. Risks & mitigations

- **A capture bug bricks voice.** → Phases 1–2 keep the Python path as a live
  fallback; the service is a child process whose death triggers fallback, not a
  crash.
- **Cross-platform audio quirks (WASAPI exclusive mode, device hot-swap).** →
  `cpal` abstracts most; the existing device-selection logic stays in Python and
  is passed to the service as config.
- **AEC.** v1 leaves AEC where it is (Python, or skip during the wake-only phase);
  moving the echo-canceller into the service is a Phase 2+ enhancement, not a v1
  blocker.
- **Build/CI for a second language.** → the service has its own `cargo` build +
  tests; the Python CI is unaffected. Ship a prebuilt binary so contributors
  without a Rust toolchain still run JARVIS.

## 7. Decision needed from the owner

1. **Rust vs Go** — recommendation is Rust (above). One word unblocks Phase 1.
2. Whether F2-in-Python (Phase 0) is enough on its own — measure first; M1 may be
   optional if the neural detector in Python already reclaims the wake cost.

Until then, Phase 0 (PR #3) delivers the felt win with no native code and no risk
to the daily-driver.
