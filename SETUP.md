# JARVIS Setup Guide

Getting JARVIS running on a fresh Windows machine. The core is quick; the
optional integrations are pick-and-choose — skip anything you don't have.

## 1. Prerequisites

- **Windows 10/11** and **Python 3.12+** (developed on 3.14). `python --version`.
- A microphone and speakers (for voice). Optional: a CUDA GPU makes local
  Whisper much faster; without one it falls back to CPU.
- An **Anthropic API key** (https://console.anthropic.com/) — the default brain.

## 2. Install dependencies

```powershell
cd <where-you-cloned-jarvis>
python -m pip install -r requirements.txt
```

`requirements.txt` marks heavy/optional packages with `# optional:` — those are
not auto-installed. The core (speech, LLM, TTS) works without them; install
extras only when you want the feature that needs them.

> **Torch/CUDA note:** `requirements.txt` pins a CUDA build of PyTorch. If that
> wheel doesn't match your GPU/CUDA, install the right one for your system from
> https://pytorch.org/get-started/locally/ (or the CPU build), then re-run the
> install. JARVIS runs on the CPU build too, just slower.

## 3. Configure

Quickest path — the guided **setup wizard** writes `.env` (your API key) and
`data/user_settings.json` (the high-impact toggles):

```powershell
python tools/setup_wizard.py
```

Or configure by hand — copy the template and fill in at least your Claude key
and your name:

```powershell
copy .env.example .env
notepad .env
```

Every variable is documented in [`.env.example`](.env.example). At minimum set:

- `ANTHROPIC_API_KEY` — required (or install Ollama for a local fallback, below).
- `JARVIS_USER_NAME` — what JARVIS calls you.

You can instead set these as **Windows User environment variables** (then open a
fresh terminal). JARVIS reads the OS environment either way.

## 4. First run

```powershell
python bobert_companion.py
```

It loads Whisper + all skills, then reaches a listening standby. Try:

| Say | Expect |
|---|---|
| "JARVIS, what time is it" | the current time |
| "JARVIS, what version are you on" | `1.88.0` |
| "JARVIS, give me a system status report" | live CPU/RAM/GPU stats |

Prefer to try it **without** using your mic/speakers? Boot the muted, mic-less
staging instance and drive it from a script:

```powershell
$env:JARVIS_STAGING="1"; $env:MUTE_TTS="1"; python bobert_companion.py --staging
# in another terminal, with ANTHROPIC_API_KEY set:
python tools/staging_integration.py -v
```

## 5. Optional services

- **Ollama** (local LLM fallback + local vision/embeddings):
  `winget install Ollama.Ollama`, then `ollama pull qwen2.5:14b-instruct` (or a
  smaller model on low-VRAM machines — the selector falls back automatically).
  Pin one with `JARVIS_LOCAL_LLM_MODEL`.
- **ComfyUI** (local image generation): run it on its default
  `http://localhost:8188`; JARVIS auto-detects it.
- **Wake word** ("Hey JARVIS"): set `PORCUPINE_ACCESS_KEY` (free at
  https://console.picovoice.ai/). Otherwise use the inject path / push-to-talk.
- **Realtime voice / neural wake** (optional, low-latency): set
  `JARVIS_VOICE_MODE=realtime` and/or `JARVIS_WAKE_WORD_AUTOSTART=1`. These need
  the optional `RealtimeSTT` / `RealtimeTTS` / `openwakeword` packages and fall
  back automatically if absent.

## 6. Optional integrations

Each is off until you provide its credential — see `.env.example`. Most have a
one-time auth step you run from a terminal (not by voice):

- **Bambu 3D printer:** set `BAMBU_PRINTER_IP` / `BAMBU_ACCESS_CODE` /
  `BAMBU_SERIAL` from the printer's LAN-Only screen.
- **Smart home** (Hue, Kasa, LIFX, Govee, Nest, Ecobee, Ring…): install the
  relevant optional library and run that skill's first-run auth, e.g.
  `python -m skills.sh_ring`. Govee just needs `GOVEE_API_KEY`.
- **OBS:** enable the WebSocket server in OBS and set `OBS_PASSWORD`.
- **Phone bridge:** set a Telegram bot token + your user id (or ntfy/Pushover).
- **Voice ID:** say "JARVIS, learn my voice" to enroll a voiceprint (stored
  locally under `data/voiceprints/`, gitignored).

## 7. Troubleshooting

- **It can't think / LLM errors:** check `ANTHROPIC_API_KEY` is set in the same
  environment you launched from, or install Ollama for a local fallback.
- **No/garbled audio:** check your default mic/speaker; see the boot log's
  device line. Heavy console output is normal.
- **A skill says "not installed/configured":** that's graceful degradation —
  install its optional dependency or set its credential to enable it.
- **Something's broken:** say "JARVIS, what's broken" — the self-diagnostic loop
  tracks failing components.

Found a real bug? See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 8. Updating

JARVIS checks GitHub for a newer release on boot and speaks a heads-up when one
is available, and you can ask **"JARVIS, check for updates"** any time. For a
**private** repo, set `JARVIS_GITHUB_TOKEN` (a fine-grained token with read-only
**Contents** access) so the check can reach the Releases API.

To apply an update:

```powershell
python tools/update_wizard.py        # check -> fast-forward -> re-run tests -> restart
```

It only fast-forwards (never clobbers local changes), re-runs the test gate, and
prints an exact rollback command if anything looks off.
