# JARVIS — a local-first voice assistant for your PC

JARVIS is an Iron-Man-style voice assistant that runs on your own Windows
machine. It listens, thinks with Claude (or a local LLM), speaks back, and can
actually *do* things — control your PC, run a battery of skills (briefings,
timers, smart-home, 3D-printer monitoring, and ~70 more), and remember context
across conversations.

```
  mic → Whisper (STT) → Claude / Ollama (LLM) → edge-tts (TTS) → speakers
        cameras → face/presence awareness        + ~78 dynamically-loaded skills
        persistent memory across conversations   + a self-diagnostic loop
```

> **Status: `1.58.0`.** This is a personal project shared for
> others to try. It's Windows-focused, expects some setup (your own API keys,
> optional hardware), and is provided as-is. Expect rough edges — and please
> file issues.

---

## What it can do

- **Conversational voice control** — natural speech in, spoken answers out, with
  an in-character "JARVIS" persona and tone-aware delivery.
- **PC control** — open apps and URLs, search, take screenshots, click/type, focus
  windows, run code snippets.
- **~78 skills** (auto-loaded from `skills/`), including: morning/evening
  briefings, weather & news, timers & reminders, a notification triage engine,
  smart-home control (Hue, Kasa, LIFX, Govee, Nest, Ecobee, …), Bambu Lab
  3D-printer monitoring, Microsoft Teams/email helpers, OBS control, pattern
  learning, and a self-diagnostic + self-healing loop.
- **Memory** — remembers facts and context across sessions (stored locally).
- **Local-first & private** — your conversations and memory stay on your machine;
  the only cloud call is to the LLM/TTS providers you configure.
- **Graceful degradation** — every integration is optional; missing a key, a
  device, or a dependency produces a friendly message, never a crash.

See [`FEATURES.md`](FEATURES.md) for the full capability list.

---

## Quickstart

Requires **Python 3.12+** (developed on 3.14) on **Windows 10/11**. A CUDA GPU is
recommended for fast local Whisper but not required.

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Configure — guided wizard (writes .env + data/user_settings.json):
python tools/setup_wizard.py
#    ...or by hand: copy .env.example .env  then fill in at least ANTHROPIC_API_KEY

# 3. Run
python bobert_companion.py
```

The first boot loads the speech model and all skills, then reaches a listening
standby. Say **"JARVIS, what time is it"** to test the round-trip.

A safe, non-intrusive way to try it without using your mic/speakers:

```powershell
$env:JARVIS_STAGING="1"; $env:MUTE_TTS="1"; python bobert_companion.py --staging
```

Full setup (optional services like Ollama / ComfyUI, smart-home auth, wake word)
is in **[`SETUP.md`](SETUP.md)**. Every configurable key is documented in
**[`.env.example`](.env.example)**.

---

## Updating

JARVIS checks GitHub for a newer release on boot and speaks a one-line heads-up
when one's available (set `JARVIS_GITHUB_TOKEN` so it can reach a private repo).
To apply an update, run the wizard — it fast-forwards, re-runs the test gate, and
tells you to restart:

```powershell
python tools/update_wizard.py          # --check to just look, --yes to skip the prompt
```

---

## Architecture (one minute)

- **`bobert_companion.py`** — the entrypoint and main loop (capture → LLM →
  parse actions → run → speak). A large monolith by design.
- **`core/`** — reusable modules extracted from the monolith: LLM client, TTS,
  memory, mode routing, voice/emotion, the action registry, etc.
- **`skills/`** — ~78 self-contained skills. Each defines `register(actions)` and
  is loaded dynamically at boot. Drop a new `.py` in here to teach JARVIS a trick.
- **`hud/`** — optional on-screen HUD overlays (tkinter / PyQt).
- **`tools/`** — dev tooling: test runner, codebase auditor, coverage, the
  **Settings GUI** (`settings_window.py`), and the setup / update wizards.
- Optional **realtime streaming voice + neural wake-word** (flag-gated via
  `JARVIS_VOICE_MODE` / `JARVIS_WAKE_WORD_AUTOSTART`), a **system-tray** applet,
  and on-screen HUD overlays.

---

## Testing

JARVIS ships with a substantial test suite (**~2,100 tests**, stdlib `unittest`,
no pytest) split into two tiers:

```powershell
python tools/run_tests.py            # fast unit suite (skills load in isolation, all I/O mocked)
python tools/run_tests.py -v         # verbose
python tools/run_coverage.py         # coverage report (core/ + skills/ + tools/)
python tools/audit_codebase.py       # static auditor (must be 0 findings)
```

- **Light tier** (the above) runs anywhere — every skill is loaded in isolation
  with mocked I/O, so no hardware/network/keys are needed. This is what CI runs.
- **Heavy/local tier** boots a real (muted, mic-less) staging instance and drives
  the end-to-end pipeline: `python tools/staging_integration.py` (needs
  `ANTHROPIC_API_KEY`).

CI (GitHub Actions, `.github/workflows/ci.yml`) runs compile + lint + the unit
suite + the auditor + a coverage floor on every push.

---

## Contributing / testing it

Bug reports and skill contributions are welcome — see
**[`CONTRIBUTING.md`](CONTRIBUTING.md)** for how to run the suite, add a skill,
and what to include in an issue.

## License

MIT — see [`LICENSE`](LICENSE). (Set the copyright holder to your name/handle
before publishing your fork.)

## Disclaimer

A hobby project, not an official product. It can control your PC and call
external APIs you configure — review what each skill does before enabling it.
No affiliation with Marvel or any "J.A.R.V.I.S." trademark; the name is an
homage.
