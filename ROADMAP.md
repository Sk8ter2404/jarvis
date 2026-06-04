# JARVIS Roadmap

Grounded in a four-dimension survey of the live tree (architecture, capabilities,
performance, self-upgrade safety). Buckets follow the project's versioning scheme:

- **`1.0.0` — Major:** big differences (architecture, paradigm, language/runtime).
- **`0.1.0` — Feature:** new capabilities.
- **`0.0.1` — Fix:** bug fixes, debt, polish.

Status: ☐ todo · ◐ in progress · ☑ done.

---

## Where it stands today

~101K-LOC local-first Windows voice assistant. `mic → Whisper → Claude emits
[ACTION: …] → ~150 handlers → edge-TTS`. Public `v1.22.0` (all releases kept,
`v1.0.0-beta.1` → `v1.22.0`); the self-upgrade pipeline's internal CHANGELOG
counter (~v1.0.17) is a SEPARATE axis. **100% unit-test coverage,
CI-green.** Genuinely strong: mature voice stack, cloud-optional (full local
fallback on a 3090), dual-store memory + personal-files RAG, a rich proactive
layer (ambient listen, "Chappie" silent learner, briefings, anticipation), the
Bambu 3D-printer companion, and self-rewriting via a multi-agent pipeline.

Central tension: a feature-rich, incident-hardened system wrapped around a
**~15K-line mutable-global monolith an LLM edits ~20×/day** — and the two things
felt most (latency, self-upgrade safety) have the clearest gaps.

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
- **DECIDED: Rust.** Scaffold landed — `native/jarvis-audio/` (the IPC protocol
  + a buildable service skeleton, `cargo test` green). Capture (cpal) + wake/VAD
  + the named-pipe transport are the next increments, additive/shadow-mode.

### M2 · De-monolith via process isolation  ☐
The monolith + ~17 shared global-state slots + ~30 `global`-rebound singletons +
~50 JSON files as IPC are the core maintainability liability — and the worst
shape for a file an LLM rewrites daily.
- Lean on the **proven** blue/green handoff (`data/handoff.json` already survives
  full process replacement mid-conversation): split into cooperating processes
  (native audio service, Python brain, vision worker, existing HUD) over a real
  IPC bus instead of 50 racy JSON files.
- Formalize `skill_utils` into a typed service interface; retire `global`
  rebinds cluster-by-cluster via the `core/actions._bc()` extraction seam.

### M3 · Trustworthy self-evolution  ◐  ([PR: feat/safe-self-upgrade])
The pipeline edits its own 101K-LOC brain unattended; the safety gaps were the
scariest finding. The 100% test suite is the missing correctness gate.
- ☑ **Risk-score actually gates** — `risk_score ≥ JARVIS_PIPELINE_MAX_RISK`
  (default 7) escalates to reject+rollback instead of being advisory.
- ☑ **Fail closed** — reviewer infra-error / unparseable JSON now score 8 and
  roll back, instead of defaulting to `approve_with_warnings` (fail-open).
- ☑ **Correctness gate** — after the boot smoke, run the unit suite; a change
  that breaks a tested contract rolls back even though it booted.
- ☐ **Git-based rollback** — replace lossy file-copy snapshots (5 kept, curated
  allow-list that misses `adapters/`) with `git` stash/commit (atomic, complete).
- ☐ **Optional human approval queue** for high-risk / core-file edits.
- ☐ **Per-run USD budget cap** (DeepAudit has one; the main pipeline does not —
  last run was $413.90).
- ☐ **Refuse autonomous runs when safety nets are disabled** (`STABILITY_GATE_DISABLE`
  / `JARVIS_PIPELINE_SKIP_TESTER` currently silent).

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
- **F4 · Wire the dead orchestrator sub-agents** — `calendar_scanner` is inert
  until a calendar skill registers `calendar_today`; un-pin the stale Haiku ID. ☐
- **F5 · Expose JARVIS as an MCP server** (it's already an MCP *client*) so other
  agents can call its ~150 skills. ☐
- **F6 · Smart-home breadth** — finish the half-abandoned Alexa integration,
  thicken Tuya, add Sonos/Roku/SmartThings. ☐
- **F7 · True eye-tracking** (current gaze is coarse left/right camera geometry). ☐

---

## 🟢 0.0.1 — Fix / debt / polish

- ☐ `skills/holographic_overlay/hud_v2.py` dead `_HAS_PYQT6` fallback — `NameError`s
  if PyQt6 is absent instead of degrading. *(spin-off task filed)*
- ☑ `core/smart_home_router.py` `best` dead-store — was an unimplemented
  "tie-for-best" feature; replaced with a working room+type fan-out (in the
  coverage PR), now tested.
- ☐ Dedupe the failure-marker lists (`bobert_companion.py` vs `dispatcher.py` —
  different strings = drift risk).
- ☐ Collapse the 3 routers' duplicated `set_state`/`conversation_history`
  bookkeeping (~8 sites).
- ☐ Delete the ~8 retired HUD overlays shipping as dead code.
- ☐ Harden CUDA DLL registration (silently regresses to CPU); fix camera-probe
  blocking 20–30s on a missing/busy cam.
- ☐ Move ~50 `*_state.json` files out of the project root into `data/`.
- ☑ `FEATURES.md` skill/HUD counts corrected (78 skills / 10 HUDs).

---

## Suggested sequence

1. **F1/F2** (realtime + neural wake) — biggest felt win, low effort (already built).
2. **M3** (safe self-evolution) — *started*; finish git rollback + approval + budget.
3. **M1** (native audio service) — the latency finale, once F1/F2 prove the path.
4. **M2** (process isolation) — the structural finale.
5. **M4** (productize) + the 0.0.1 polish, ongoing.
