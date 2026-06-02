# Design: M2 — De-monolith via process isolation

Status: **proposal** · Roadmap item: `1.0.0` / M2 · Author: design pass 2026-06-02

> The structural finale. Dissolve the 14.7K-line monolith — but by *process
> isolation along seams the system already uses*, not a risky in-place rewrite of
> a file an LLM edits ~20×/day.

---

## 1. Problem

`bobert_companion.py` is ~14.7K lines / 265 top-level defs and still owns audio
capture, STT, TTS/lipsync, vision, UI automation, window control, the LLM
dispatcher, the skill loader, the tray bridge, blue/green, and the boot+loop. Three
coupled liabilities (from the architecture survey):

1. **Global mutable state, load-bearing.** ~17 `core/state.py` single-element-list
   slots + ~30 `global`-rebound monolith singletons + `conversation_history` +
   `ACTIONS`, shared lock-free across dozens of daemon threads and 78 skills via
   `from core.state import *` and `bc._x[0]`. This is the single biggest blocker to
   parallelism, process-splitting, and testability (the test harness needs a
   `_JARVIS_SINGLETON_PID` sentinel + a `MonolithGlobalsTestCase` that
   snapshots/deep-restores ~70 globals after *every* test purely to compensate).
2. **File-as-IPC sprawl.** Tray↔monolith, HUD↔monolith, blue/green handoff,
   injected commands, pending speech, timers — all flow through ~50 JSON files in
   the project root (mitigated by `core/atomic_io` + locks, but ad-hoc and racy).
3. **An LLM rewrites this file daily.** A giant mutable-global module is the worst
   possible target for the self-upgrade pipeline — every diff is against 690 KB and
   risks cross-cutting breakage.

## 2. The key insight — the escape hatch already exists

JARVIS **already survives a full process replacement mid-conversation.** The
blue/green path boots a second process and hands off the in-flight conversation
tail + timers via `data/handoff.json` (`_consume_blue_green_handoff`). That is
*proof* the natural seam here is **process boundaries, not module boundaries** —
and that the GIL-atomic single-element-list contract (which breaks under
subinterpreters / free-threading) can be sidestepped rather than fought.

## 3. Goals / non-goals

**Goals**
- Split the monolith into cooperating **processes** along the seams the system
  already exposes, each independently restartable and testable.
- Replace the ~50-file IPC with one structured **message bus**.
- Give skills a **typed capability interface** instead of a dict of lambdas
  closing over monolith globals.
- Shrink the LLM-edited surface so a self-upgrade touches one small service, not a
  690 KB god-module.

**Non-goals**
- A from-scratch rewrite. This is incremental extraction; the monolith shrinks
  process-by-process and never stops booting.
- Changing the skill *contract* (`register(actions)` + `[ACTION: name, arg]`
  dispatch stays; only the *plumbing* under `skill_utils` changes).
- Touching `core/state.py`'s single-element-list idiom *within* a process — it's
  correct and stays; it just stops being shared *across* the new boundaries.

## 4. Target architecture

```
                         ┌──────── message bus (local sockets / named pipes) ────────┐
                         │                                                            │
┌──────────────┐   ┌─────┴──────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┴───┐
│ audio svc    │   │  BRAIN     │   │ vision svc   │   │  HUD (exists)│   │  tray (exists)│
│ (M1, native) │──▶│ (Python)   │◀─▶│ (Python)     │   │  subprocess  │   │  subprocess   │
│ capture/wake │   │ loop+LLM+  │   │ camera/face/ │   │              │   │               │
│ /VAD         │   │ skills+mem │   │ screen-VLM   │   │              │   │               │
└──────────────┘   └────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

- **BRAIN** (Python) keeps the conversation loop, LLM dispatch, `ACTIONS`, memory,
  orchestrator — the parts that are inherently sequential + Python. It owns the
  authoritative `RuntimeState`.
- **audio svc** is M1 (native; see [M1-native-audio-service.md](M1-native-audio-service.md)).
- **vision svc** lifts the 20 FPS OpenCV face loop + screen-VLM out of the brain's
  GIL into its own process; publishes `gaze`/`face`/`screen_summary` events.
- **HUD** and **tray** are *already* separate subprocesses — they just move from
  JSON-file polling to subscribing on the bus.

### The message bus

One structured local IPC layer (named pipes on Windows / UDS elsewhere) carrying
typed messages: a request/response channel (brain↔service RPC) + a pub/sub channel
(events: `wake`, `gaze`, `print_progress`, `state_changed`, …). Replaces the ~50
`*_state.json` files. Keep it dead-simple: length-prefixed JSON for control,
binary frames only where latency demands (audio PCM). One schema module shared by
all processes.

### The two seams that make this incremental

1. **`skill_utils` → a typed `JarvisServices` interface.** Today `skill_utils` is a
   dict of ~15 lambdas closing over monolith globals (`ask_vision`, `take_screenshot`,
   `click`, `write_hud_state`, promise helpers). Formalize it into a small typed
   object whose methods are RPCs to the owning service. Skills depend on the
   *interface*, not the module — so a skill doesn't care whether vision is in-process
   or across the bus. **This is the highest-leverage refactor and can land first,
   independent of any process split** (it immediately improves testability).
2. **`core/actions._bc()` late-bind → explicit state.** Handlers already reach the
   monolith via a deferred `bc = _bc()`. Continue migrating handler clusters (audio,
   vision, TTS, window) into `core/` modules that take an explicit `RuntimeState`
   argument, retiring `global` rebinds one cluster at a time. The
   `core/state.py:contract` already spells out the lockstep hazard — respect it:
   every skill's `bc._x[0]` access moves in lockstep with the slot it reads.

## 5. Migration plan (each phase ships green, monolith always boots)

- **Phase 1 — typed services interface (no process split).** Introduce
  `core/services.py` (`JarvisServices` protocol) and back it with the *current*
  in-process implementations. Migrate `skill_utils` consumers to it. Pure
  refactor; massive testability win; zero behavior change. *Lands first.*
- **Phase 2 — the bus, in-process.** Add the message-bus abstraction with an
  in-process transport (a queue) as the default. HUD/tray switch from JSON-file
  polling to bus subscriptions (file IPC retained as fallback for one release).
- **Phase 3 — lift vision out.** Move the camera/face/screen loop into a vision
  service process behind the bus. The brain's GIL stops contending with the 20 FPS
  OpenCV loop. Behind `VISION_SERVICE_ENABLED` (default False) with the in-process
  path as fallback.
- **Phase 4 — audio service (= M1).** The native capture/wake/VAD service joins the
  bus. At this point the brain no longer owns the mic.
- **Phase 5 — shrink the brain.** With audio/vision/HUD/tray external, the monolith
  is now "the loop + LLM + skills + memory." Split *that* file by theme into
  `core/` modules behind the typed services (the boot/loop stays the entrypoint).

The self-upgrade pipeline benefits at every phase: its edit surface shrinks from
one 690 KB god-module toward small, single-responsibility services.

## 6. Risks & mitigations

- **Cross-process state races.** → one **authoritative owner per state slot** (the
  brain owns `RuntimeState`; services hold read-replicas updated by events). No
  more "any module mutates any flag." This is *stricter* than today, not looser.
- **Latency of bus round-trips for hot calls.** → keep genuinely-hot paths
  in-process (the brain's loop); only cross the bus for naturally-async work
  (vision events, HUD updates, audio frames).
- **A big-bang refactor stalls.** → Phase 1 (typed interface) delivers most of the
  testability win with *zero* process split and is independently valuable, so the
  effort front-loads safety and can pause between phases.
- **Debuggability of N processes.** → the bus logs every message to one structured
  stream; `self_diagnostic` already probes subsystems — extend it to the bus.

## 7. Decision needed from the owner

1. **Commit to Phase 1 first** (the `JarvisServices` typed interface) — it's the
   highest-value, lowest-risk slice and unblocks everything else. Recommend yes.
2. Bus transport detail (named-pipe vs UDS-on-Windows-via-AF_UNIX) — a Phase-2
   implementation choice, not needed now.

This is the largest item on the roadmap; it should land **after** M1/M3 and only
in the phased, always-bootable form above — never as a stop-the-world rewrite.
