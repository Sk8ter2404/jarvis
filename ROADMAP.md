# JARVIS Roadmap

Grounded in a four-dimension survey of the live tree (architecture, capabilities,
performance, self-upgrade safety). Buckets follow the project's versioning scheme:

- **`1.0.0` — Major:** big differences (architecture, paradigm, language/runtime).
- **`0.1.0` — Feature:** new capabilities.
- **`0.0.1` — Fix:** bug fixes, debt, polish.

Status: ☐ todo · ◐ in progress · ☑ done.

---

## Where it stands today

*Status lines refreshed 2026-10-02 (v2.0.173); the survey text below them is
from 2026-07-09.*

~530K lines of tracked Python (~230K outside `tests/`), local-first Windows
voice assistant. `mic → Whisper (Parakeet optional) → the local Ollama brain or
Claude emits [ACTION: …] → ~440 handlers (705 action names) → edge-TTS /
Kokoro`. Release `VERSION` 2.0.173, every release tagged from
`v1.0.0-beta.1`; the self-upgrade pipeline's internal CHANGELOG counter is a
SEPARATE axis. CI runs the light test tier on Linux and gates coverage at 80%.
Genuinely strong: mature voice stack, cloud-optional (the local model answers
every turn when chat routing is local or there is no key), dual-store memory +
personal-files RAG, a rich proactive layer (ambient listen, "Chappie" silent
learner, briefings, anticipation, phone pings), the Bambu 3D-printer
companion, and self-rewriting via a multi-agent pipeline. Feature list:
[FEATURES.md](FEATURES.md).

Central tension: a feature-rich, incident-hardened system wrapped around a
**~43K-line mutable-global monolith** — and the two things felt most (latency,
self-upgrade safety) have the clearest gaps.

**Recently shipped (1.1 → 1.3):** realtime voice + neural wake flag-wired (F1/F2),
the tray overhaul + standalone Settings GUI, the M3 self-upgrade safety gates, an
**update checker** (boot nudge + the `check for updates` action), an **update
wizard**, a first-run **setup wizard**, an automatic **PII pre-commit guard**, and
M2 Phase 2 groundwork — the in-process **message bus** (`core/message_bus.py`,
unwired) — plus the **M1 native-audio-service scaffold** (`native/jarvis-audio/`,
Rust: the IPC protocol + a buildable service skeleton, cargo-tested; capture /
wake / VAD next). (Heading range above: 1.1 → 1.6.)

---

## 🔴 1.0.0 — Major

### M1 · Streaming, sub-second hot path  ◐
Today: two serial un-streamed network round-trips per turn (Claude + edge-TTS) +
a fixed 1.4s silence endpoint → **3–5s** felt latency. Worse, standby runs a
**full GPU Whisper inference per utterance just to substring-match "jarvis"**
while a real neural detector (`core/wake_word.py`) sits unused.
- Native (Rust/Go) always-on **audio + wake + VAD** service; hand only post-wake
  PCM to Python. Kills the Whisper-as-wake waste + the audio-path GIL contention.
- Streaming STT→LLM→TTS so the first syllable plays before the reply completes.
- *Down payment available now as F1/F2 (flip on the already-built realtime mode).*
- **DECIDED: Rust.** `native/jarvis-audio/` has the IPC protocol, cpal capture,
  a ring buffer and VAD/endpointing; wake word + the named-pipe transport are
  next, and nothing in Python uses it yet.
- ◐ In the Python pipeline meanwhile (the "speed plan", each step off or
  shadow-only until measured): per-turn `[turn-timing]` telemetry, Parakeet STT
  on the CPU with a shadow A/B mode, Whisper decode knobs, Smart Turn
  end-of-turn (shadow), Kokoro phonemizer + render cache, and a cache-stable
  local prompt with idle re-prime.

### M2 · De-monolith via process isolation  ◐ (groundwork only, unwired)
The monolith + ~17 shared global-state slots + ~30 `global`-rebound singletons +
~50 JSON files as IPC are the core maintainability liability — and the worst
shape for a file an LLM rewrites daily.
- Lean on the **proven** blue/green handoff (`data/handoff.json` already survives
  full process replacement mid-conversation): split into cooperating processes
  (native audio service, Python brain, vision worker, existing HUD) over a real
  IPC bus instead of 50 racy JSON files.
- Formalize `skill_utils` into a typed service interface; retire `global`
  rebinds cluster-by-cluster via the `core/actions._bc()` extraction seam.
- Built, not yet wired: the in-process bus (`core/message_bus.py`), its
  cross-process wire format (`core/bus_transport.py`) and the typed skill
  seam (`core/services.py`).

### M3 · Trustworthy self-evolution  ◐  ([PR: feat/safe-self-upgrade])
The pipeline edits its own 101K-LOC brain unattended; the safety gaps were the
scariest finding. The 100% test suite is the missing correctness gate.
- ☑ **Risk-score actually gates** — `risk_score ≥ JARVIS_PIPELINE_MAX_RISK`
  (default 7) escalates to reject+rollback instead of being advisory.
- ☑ **Fail closed** — reviewer infra-error / unparseable JSON now score 8 and
  roll back, instead of defaulting to `approve_with_warnings` (fail-open).
- ☑ **Correctness gate** — after the boot smoke, run the unit suite; a change
  that breaks a tested contract rolls back even though it booted.
- ☐ **Git-based rollback** — still file-copy snapshots (now of the wider code
  surface each task can touch, under `backups/pipeline/`), not `git`.
- ☐ **Optional human approval queue** for high-risk / core-file edits.
- ☑ **Per-run USD budget cap** — `JARVIS_PIPELINE_MAX_USD`, default $25.
- ☑ **Refuse autonomous runs when safety nets are disabled** — an autonomous
  run with `STABILITY_GATE_DISABLE`, `JARVIS_PIPELINE_SKIP_TESTER` or
  `JARVIS_PIPELINE_SKIP_SUITE` set does not start without an explicit
  `--force-unsafe`.
- The overnight engine ships paused (`OVERNIGHT_UPGRADE_ENABLED = False`).

### M4 · Productize for distribution  ◐
Today it's "clone + pip + GPU + many env vars" into **global Python 3.14** (~7GB
wheels, fragile CUDA DLL registration, missing 3.14 wheels). → venv + pinned
deps + frozen installer + a no-GPU/CPU-fallback profile.
- ☑ Shipped: first-run **setup wizard**, **Settings GUI**, **update checker +
  update wizard**, **PII pre-commit guard** — directly addressing the
  env-var / onboarding pain. Remaining: venv + pinned deps + frozen installer +
  a CPU-fallback profile.

---

## 🟡 0.1.0 — Feature

- **F1 · Realtime streaming voice** (`core/realtime_voice.py`) — built + flag-wired
  via `core/voice_pipeline.py` (`JARVIS_VOICE_MODE=realtime`), default-off, safe
  fallback. The native always-on service (M1) is still the bigger latency win. ◐
- **F2 · Neural wake detector** (`core/wake_word.py`) — built + flag-wired
  (`JARVIS_WAKE_WORD_AUTOSTART`), default-off; stops paying Whisper to listen for
  one word once enabled. ◐
- **F3 · Real Claude tool-use API** instead of `[ACTION: name, arg]` text parsing
  (fewer hallucination guards, more robust). ☐
- **F4 · Wire the dead orchestrator sub-agents** — `calendar_today` /
  `calendar_next` now exist (`skills/ms_graph.py`) and the models are current
  (Sonnet 5.5 planner/merger, Haiku 4.5 workers, 2026-10-01); the fan-out runs
  only when the cloud is allowed. ☑
- **F5 · Expose JARVIS as an MCP server** (it's already an MCP *client*) so other
  agents can call its ~700 actions. ☐
- **F6 · Smart-home breadth** — finish the half-abandoned Alexa integration,
  thicken Tuya, add Sonos/Roku/SmartThings. ☐
- **F7 · True eye-tracking** (current gaze is coarse left/right camera geometry). ☐

---

## 🟢 0.0.1 — Fix / debt / polish

- ☑ `skills/holographic_overlay/hud_v2.py` `_HAS_PYQT6` fallback — the Qt base
  classes are stubbed to `object` when PyQt6 is absent, so it imports and
  `main()` exits 2.
- ☑ `core/smart_home_router.py` `best` dead-store — was an unimplemented
  "tie-for-best" feature; replaced with a working room+type fan-out (in the
  coverage PR), now tested.
- ☑ Dedupe the failure-marker lists — one list, `core/failure_markers.py`,
  shared by `bobert_companion.py` and `core/dispatcher.py`.
- ☐ Collapse the 3 routers' duplicated `set_state`/`conversation_history`
  bookkeeping (~8 sites).
- ◐ Retired HUD overlays — every `hud/` overlay but one has a launcher; the old
  corner ring `hud/jarvis_hud.py` still ships with nothing launching it.
- ☐ Harden CUDA DLL registration (silently regresses to CPU); fix camera-probe
  blocking 20–30s on a missing/busy cam.
- ☐ Move the `*_state.json` files out of the project root into `data/` (24
  still at the root of a running install, 2026-10-02).
- ☑ `FEATURES.md` rewritten against the tree (2026-10-02): 99 skills, 705
  action names, local-first routing; its counts table says how each number
  was measured.

---

## Suggested sequence

1. **F1/F2** (realtime + neural wake) — biggest felt win, low effort (already built).
2. **M3** (safe self-evolution) — *started*; budget cap done, git rollback +
   approval queue left.
3. **M1** (native audio service) — the latency finale, once F1/F2 prove the path.
4. **M2** (process isolation) — the structural finale.
5. **M4** (productize) + the 0.0.1 polish, ongoing.
