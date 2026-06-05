---
name: run-jarvis
description: Build, launch, and DRIVE the JARVIS (Bobert) voice assistant headlessly — boot it, inject voice commands without a mic, read its spoken replies, screenshot the HUD, and run the test suite. Use when asked to run / start / launch / boot / drive / smoke-test / screenshot JARVIS or Bobert, or to verify a change against the live app.
---

# Run JARVIS (Bobert)

JARVIS is a ~100K-LOC **Python 3.14 Windows** voice assistant — one monolith
(`bobert_companion.py`) plus ~77 skills. It normally listens on a mic, but it
has a file-based **inject channel** (`injected_commands.json`, drained at the
top of every main-loop iteration — the PROD path, not a test stub) so you can
drive the **live app with no microphone**.

**Drive it with the committed driver — this is the agent path:**

```
python .claude/skills/run-jarvis/driver.py "what time is it"
```

The driver boots JARVIS if it isn't running, force-wakes it out of standby,
injects the utterance, then tails the session log and prints JARVIS's reply +
the `[action]` result. All paths below are relative to the project root
(the dir holding `bobert_companion.py`, e.g. `C:\JARVIS`).

> Platform: this is a **Windows** app driven from **PowerShell** (`pythonw`,
> the tray, the HUD, `pygetwindow`, media keys are all Win32). It is not a
> Linux app; there are no `apt-get` steps.

## Prerequisites

- **Windows 10/11**, **Python 3.14** (`py -3.14` / the `pythoncore-3.14` env).
- **Ollama** running with a local model pulled, for local routing:
  `ollama pull qwen2.5:14b-instruct-q5_K_M` (the 14B that fits a 24GB GPU).
- Python deps: `python -m pip install -r requirements.txt` (numpy, sounddevice,
  faster-whisper, pygetwindow, psutil, requests, edge-tts, …). The app degrades
  gracefully when optional ones are missing (no pyaudio → turn-based voice; no
  ffmpeg → a pydub warning; no API key → local-only).
- For cloud routing (more accurate than the 14B), set an Anthropic key in the
  environment and `AI_BACKEND=claude` in `data/user_settings.json`.

## Run (agent path) — the driver

```bash
# is it up?  (heuristic: the session log was written in the last 15s)
python .claude/skills/run-jarvis/driver.py --status

# drive a command end-to-end (boots + wakes automatically if needed)
python .claude/skills/run-jarvis/driver.py "what's the weather"
python .claude/skills/run-jarvis/driver.py "play billie jean"

# (re)boot first, then drive
python .claude/skills/run-jarvis/driver.py --boot "system status"

# just wake it out of sleep/standby
python .claude/skills/run-jarvis/driver.py --wake
```

Example (verified this session — booted from cold, drove, got the reply):

```
=== JARVIS (ok) ===
[13:10:16]   JARVIS: [intent:confirmation] [ACTION: get_time]
[13:10:16]   [action] get_time: current time is 01:10 PM on Friday
[13:10:18]   JARVIS: [intent:briefing] 1:10 PM on Friday, sir.
```

**Screenshots / the HUD.** JARVIS has a `screenshot` action that writes a
full-desktop PNG — drive it and read the file:

```bash
python .claude/skills/run-jarvis/driver.py "take a screenshot"
# -> [action] screenshot: screenshot saved to <proj>\screenshots\screenshot_<ts>.png
ls screenshots/                      # newest PNG is the capture
```

That PNG is **unmasked** (it's the app's own capture), so the HUD overlay shows
up in it — handy because the HUD is a `pythonw` window that an external
screenshotter may mask. The HUD monitor is `HUD_MONITOR` in
`data/user_settings.json` (`top` / `middle` / `right` / …); it also restores its
last dragged position from `unified_hud_geometry.json`, which OVERRIDES
`HUD_MONITOR` — to relocate it, set `HUD_MONITOR` **and** overwrite that file.

## Run (human path)

```powershell
$env:JARVIS_LOCAL_LLM_MODEL = 'qwen2.5:14b-instruct-q5_K_M'   # dodge the 32B brick
& .\_boot_jarvis.ps1        # launches detached pythonw + a live-log viewer window
```

Then talk: say **"JARVIS"** to wake, then your command. Stop via the tray Quit,
voice "JARVIS, shut down", or `Stop-Process -Id <pid>`. Useless without a mic —
prefer the driver for automation.

## Test

The faithful gate (simulates the Linux/light-deps GitHub CI — run this before
pushing, NOT the bare `run_tests.py`, which assumes dev-box deps):

```bash
python tools/run_tests_ci_sim.py        # ~12,200 tests, ~35s; expect 0 failed / 0 errored
```

The monolith IS unit-testable via an import harness (`tests/monolith/`,
`load_monolith()` + a singleton sentinel); skills load in isolation
(`tests/skills/`, `load_skill_isolated()`). Set `JARVIS_SETTINGS_PATH` to a temp
file when running tests directly so they can't clobber the live
`data/user_settings.json`.

## Gotchas (the battle scars)

- **Standby SILENTLY DROPS injected commands.** If JARVIS has dozed
  (`[standby] ignored: '<your text>'` in the log), the inject is gone. The wake
  WORD only comes from the mic, so it can't wake the inject path — you must
  **force-wake via the tray channel**: write `[{"cmd":"force_wake"}]` to
  `tray_commands.json` (the driver does this automatically and retries once).
  It dozes on idle and on mis-hearing a sleep phrase, so re-wake per batch.
- **The 32B local model BRICKS a 24GB GPU.** `qwen2.5:32b` (~22 GB) + on-demand
  vision (~7 GB) + whisper (~1.5 GB) blows past 24 GB → 50 s timeouts and "both
  local and cloud unavailable". A model-resolver preference chain prefers the
  biggest installed model over your setting, so **force the 14B** with
  `JARVIS_LOCAL_LLM_MODEL=qwen2.5:14b-instruct-q5_K_M` on boot (the driver sets
  this). Or route chat/vision to cloud (`AI_BACKEND=claude`).
- **Smart App Control (SAC) can block Ollama's `ggml.dll`** (CodeIntegrity event
  3077): `/api/tags` still 200s but generation fails → intermittent "Ollama dead
  on boot". No per-file allowlist; only SAC-off truly fixes it (one-way).
- **`tools/say_to_jarvis.py` enqueues fine but its `print()` can crash** on a
  cp1252 console (a stray `→`/`…`). The command IS queued before the crash. The
  driver writes `injected_commands.json` directly to avoid this entirely.
- **Vision-click on a multi-monitor rig needs a monitor pin.** `find_click_target`
  defaults to photographing the whole virtual screen, which downscales controls
  too small to find; streaming auto-play pins capture to the player window's
  monitor. Negative-origin monitors (e.g. a display at x=-2560) are handled.
- **`webbrowser.open` opens the UWP Apple Music APP, not a browser** (it
  registered as the `music.apple.com` handler). Media play forces Chrome via
  `_open_url_in_browser`; it also reuses ONE tab per service (closes the prior
  one) and double-clicks the iTunes-resolved track row to play.
- **Force-killing Chrome triggers a "Restore pages?" dialog that steals
  keystrokes.** Close browser tabs gracefully (`pygetwindow .close()`), or patch
  `Default/Preferences` `profile.exit_type=Normal` before relaunch.
- **App Control blocks running `*.exe` here.** Use `python -m pyflakes`, not
  `pyflakes.exe`; git-bash can't exec `.exe` shims.
- **An action runs but JARVIS doesn't SPEAK the result?** Its result is only
  voiced if the action name is in `INFORMATIVE_ACTIONS` (LLM re-summarises) or
  `SPEAK_RESULT_VERBATIM_ACTIONS` (spoken as-is) in `bobert_companion.py`. A new
  read-out action must be added to one of those sets or only the "Of course,
  sir" preamble is heard.
- **Only `data/`-gitignored runtime state is personal** (voiceprint, memory,
  tokens, smart-home/printer creds). The core app boots to standby and drives
  fine WITHOUT them — graceful degradation everywhere.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `driver.py --status` says False but a window is open | The log wasn't written in 15 s (deep sleep). `--wake`, or `--boot` to be sure. |
| Inject does nothing, log shows `[standby] ignored` | Force-wake first: `driver.py --wake` (the driver auto-retries once). |
| "both local and cloud unavailable" / 50 s hangs | The 32B bricked the GPU. Boot with `JARVIS_LOCAL_LLM_MODEL=qwen2.5:14b-instruct-q5_K_M`. |
| Generation fails but `ollama list` works | SAC blocked `ggml.dll` (event 3077). Restart Ollama; last resort SAC-off. |
| `run_tests.py` errors on import (numpy/sounddevice) | That's the heavy tier. Use `tools/run_tests_ci_sim.py` (the light/CI tier). |
| Standalone monolith test conflicts with the running app | The singleton lock. Use `run_tests_ci_sim.py` (sets the sentinel) or stop the live instance. |
