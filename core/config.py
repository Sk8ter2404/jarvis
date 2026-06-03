"""JARVIS module-level configuration constants.

Extracted from bobert_companion.py on 2026-05-29 as Phase 1 of the
modularisation refactor. The parent module pulls these via `from
core.config import *` at the top of its own constant block so existing
code keeps working unchanged.

Add new top-level config knobs HERE, not in bobert_companion.py. The
reviewer's context budget and the implementer's diff size both shrink
every time we move a stable group of constants out of the monolith.

Almost nothing that does I/O or non-trivial computation at import time
belongs in this file. The single exception is RAG_INDEX_PATHS, which
calls os.path.expanduser("~") so the per-user paths resolve at the
moment of import. Everything else is dumb-values-only so import stays
microsecond-fast and side-effect-free. Helpers that *consume* these
values live in their respective modules (audio, vision, etc.).
"""
import json
import os

# ─── Location / device role ────────────────────────────────────────────
# Used by multiple skills to gate behaviour by physical location.
LOCATION   = "desk"            # e.g. "desk", "bedroom", "office", "laptop"


# ─── User identity ─────────────────────────────────────────────────────
# The assistant addresses its user by this name and recognises them as the
# owner (calendar "is this me?" checks, voice-ID default, briefing
# personalisation). Set JARVIS_USER_NAME in your environment (or .env).
# Blank => a generic "User"; no personal name is ever committed to the repo.
USER_NAME = os.getenv("JARVIS_USER_NAME", "")


# ─── Robot (physical) ──────────────────────────────────────────────────
# Set ROBOT_ENABLED = False for voice-only mode (current production).
ROBOT_ENABLED = False
ROBOT_IP      = "192.168.1.XXX"   # printed on Serial at boot
ROBOT_PORT    = 80


# ─── PC control ────────────────────────────────────────────────────────
# Master switch — let the LLM launch apps, open URLs, etc.
PC_CONTROL_ENABLED = True


# ─── Screen vision ─────────────────────────────────────────────────────
# JARVIS can see and reason about what's on the screen via Claude's
# vision-capable model.
SCREEN_VISION_ENABLED = True
SCREEN_VISION_MODEL   = "claude-sonnet-4-6"


# ─── UI automation ─────────────────────────────────────────────────────
# Click, type, navigate apps autonomously (needs: pip install pyautogui pillow).
UI_AUTOMATION_ENABLED = True


# ─── Skills system ─────────────────────────────────────────────────────
# JARVIS can write new Python modules under skills/ to teach himself
# new tasks. Loaded at startup by load_skills().
SKILLS_ENABLED = True


# ─── Safety: hard confirmation keywords ────────────────────────────────
# Actions matching these always require spoken confirmation ("yes" or
# "confirm" as the next utterance) before executing. Set to [] to
# disable (NOT recommended).
CONFIRM_KEYWORDS = ["purchase", "buy", "pay", "checkout", "delete", "format", "transfer"]


# ─── Safety: JARVIS-style pushback (soft layer) ────────────────────────
# Triggers an in-character objection ("If I may, sir — that will close
# 14 windows including your unsaved Bambu Studio project. Are you
# certain?") for gray-zone actions and defers them onto the same
# _pending_confirmation queue that CONFIRM_KEYWORDS uses. Lower = more
# cautious.
PUSHBACK_ENABLED              = True
PUSHBACK_MAX_CLOSE_WINDOWS    = 5    # >N matched windows triggers pushback
PUSHBACK_MAX_QUEUE_TASKS_BULK = 10   # >N items in a single queue_task call
PUSHBACK_MAX_CLEAR_PENDING    = 10   # >N pending tasks before clear_tasks asks


# ─── Primary LLM backend ───────────────────────────────────────────────
# AI_BACKEND="claude" means: PREFER Claude when it's reachable, but the
# LOCAL Ollama model is the always-on baseline brain — JARVIS is fully
# functional with NO Claude API key / NO credits at all. Claude is a BONUS
# that sharpens replies when available, not a requirement (see CLAUDE_OPTIONAL).
AI_BACKEND   = "claude"           # "claude" | "ollama"
CLAUDE_MODEL = "claude-sonnet-4-6"
OLLAMA_MODEL = "llama3"

# Claude API is an OPTIONAL ENHANCEMENT, never a hard dependency. When True
# (the default), a missing/capped/errored Claude backend is NOT treated as a
# failure: startup does not abort, the self-diagnostic does not raise a
# high-severity alarm or queue a "fix", and JARVIS simply runs on the local
# model. Set False only if you want JARVIS to insist on a working Claude key.
# 2026-05-30, per user: "I don't want to NEED API credits — it's a bonus."
CLAUDE_OPTIONAL = True


# ─── Local-LLM baseline (3090 Ollama is the always-on brain) ───────────
# The LOCAL model is JARVIS's baseline — it serves every turn when Claude
# is unavailable (no key, capped credits, rate-limit, network glitch, 5xx)
# and the experience is meant to be good on its own. The workshop rig's
# 3090 (24 GB VRAM) runs qwen2.5:14b q5_K_M at ~10 GB. For local calls the
# giant Claude-tuned PC_CONTROL_PROMPT is swapped for a compact action
# cheatsheet (see _local_cheatsheet) and the context is capped at 16k so
# the model fits 100 % on the GPU. The runtime selector
# `_get_local_llm_model()` consults JARVIS_LOCAL_LLM_MODEL first, then walks
# a fallback chain (qwen2.5:14b → llama3.1:8b → first available tag). Vision
# queries prefer the cloud but fall back to the local qwen2.5vl:7b.
LOCAL_LLM_FALLBACK = True
LOCAL_LLM_MODEL    = "qwen2.5:14b-instruct-q5_K_M"
LOCAL_LLM_BASE_URL = "http://localhost:11434"

# When True, every ambient/background one-shot LLM call (memory extraction,
# proactive comments, the ambient extractor — everything routed through
# `_llm_quick`) runs on the LOCAL model ONLY and never touches Claude, so
# ambient learning costs $0. Foreground conversation and user-invoked briefings
# are unaffected. Default False (Claude-first with local fallback) so a
# cloud-only install still learns; set True (Settings GUI / user_settings.json)
# when a local model is available and you want ambient learning to be free.
AMBIENT_LEARNING_FORCE_LOCAL = False

# ─── Per-function model routing ────────────────────────────────────────
# Choose, PER FUNCTION, which brain answers — so e.g. screen vision can run on
# the free local VLM while chat stays on Claude. Each value is one of:
#   "auto"  — Claude when available, local model on failure (the default)
#   "local" — local Ollama model ONLY ($0, never Claude)
#   "cloud" — Claude (today same as auto; reserved for a future no-fallback mode)
# Override individual keys via user_settings.json (partial dicts MERGE over these
# defaults) or the Settings GUI, e.g. {"MODEL_ROUTING": {"vision": "local"}}.
MODEL_ROUTING = {
    "chat":    "auto",    # foreground conversation (_call_llm)
    "vision":  "auto",    # screen / camera vision (ask_vision / ask_vision_multi)
    "ambient": "auto",    # background learning (_llm_quick); AMBIENT_LEARNING_FORCE_LOCAL also forces this local
}


def model_route(function: str) -> str:
    """The configured backend route for a JARVIS function: 'auto' | 'local' |
    'cloud'. Unknown functions default to 'auto'."""
    return MODEL_ROUTING.get(function, "auto")


# ─── Ambient passive-learning toggles (skills/ambient_listen.py) ───────
# Settings-GUI / user_settings.json knobs for the passive multimodal
# daemons. These live here (not as literals in bobert_companion.py) so a
# saved override flows through `from core.config import *` to the
# skills/ambient_listen.py autostart and _get_config() reads.
#
# AMBIENT_LISTEN_ENABLED — autostart the mic-only ambient transcription
#   daemon. False by default because it competes with record_speech for
#   the input device (Windows WASAPI rejects two opens on the same mic).
# AMBIENT_SCREEN_ENABLED — autostart periodic screen-snapshot analysis via
#   the local VLM for ambient context. False by default (privacy).
AMBIENT_LISTEN_ENABLED = False
AMBIENT_SCREEN_ENABLED = False

# ─── Screenshot privacy blocklist (vision capture guard) ───────────────
# Case-insensitive substring patterns checked against the FOCUSED window
# title before ANY screen capture for vision (see_screen / ask_vision) or
# a saved screenshot. If the active window's title contains any entry,
# JARVIS refuses the capture instead of sending/saving the screen. Empty
# default = no change (opt-in); add e.g. "1password", "bitwarden",
# "banking" via the Settings GUI / user_settings.json to enforce.
SCREENSHOT_PRIVACY_BLOCKLIST: list = []

# ─── Spend ceilings (Settings GUI exposes both) ────────────────────────
# DAILY_BUDGET_USD — hard cap on Claude spend per UTC day for the Chappie
#   continuous-learning loop (skills/chappie_consciousness.py). Default
#   1.0 to match that module's prior literal.
# DEEP_AUDIT_BUDGET_USD — default daily ceiling for the background
#   deep-audit diagnostic (core/diagnostic_daemons.py). The env var
#   JARVIS_DEEP_AUDIT_BUDGET_USD still overrides this when set. Default
#   5.0 to match the daemon's prior DEEP_AUDIT_DEFAULT_BUDGET_USD.
DAILY_BUDGET_USD      = 1.0
DEEP_AUDIT_BUDGET_USD = 5.0


# ─── Sub-agent orchestrator (core/orchestrator.py) ─────────────────────
# Decompose complex requests into parallel sub-tasks dispatched to
# cheaper workers (Haiku or local Ollama) with restricted tool subsets.
# The planner reads sub-agent specs from skills/sub_agents/*.json and
# emits a JSON plan; workers run in parallel via asyncio.gather; a
# merger Claude call synthesises results into one TTS-ready reply.
#
# WIRED + VALIDATED 2026-05-31: bobert_companion._maybe_orchestrate routes
# standing-briefing requests ("morning briefing", "summarise my day", "system
# status brief", "orchestrate …" — see _ORCHESTRATE_RE) through orchestrate();
# workers fetch REAL data via the sub-agent's first registered read action and
# the LLM only summarises that (no fabrication — a sub-agent with no registered
# action returns empty and is omitted). Sub-agent specs were reconciled to
# JARVIS's real actions: email_reader→email_briefing, news_fetcher→news_briefing,
# weather_scout→weather_briefing, system_inspector→system_pulse. Live-verified
# end-to-end producing a real multi-source brief (real inbox status + current
# news headlines + weather + system). calendar_scanner gracefully omits until a
# calendar skill registers calendar_today/ms_graph_calendar.
#
# ON by default: the trigger is narrow (only standing morning/daily/system
# briefs, never an arbitrary "brief me on X"), so normal turns are untouched;
# each fired brief costs a planner + parallel-worker + merger LLM fan-out.
# Set False (or it's also gateable via JARVIS_ENABLE_ORCHESTRATOR) to disable.
ENABLE_ORCHESTRATOR             = True
ORCHESTRATOR_PLANNER_MODEL      = "claude-sonnet-4-6"
ORCHESTRATOR_WORKER_MODEL       = "claude-haiku-4-5"
ORCHESTRATOR_MERGER_MODEL       = "claude-sonnet-4-6"
ORCHESTRATOR_MAX_PARALLEL       = 4
ORCHESTRATOR_WORKER_TIMEOUT_S   = 30.0
ORCHESTRATOR_PLANNER_TIMEOUT_S  = 20.0
ORCHESTRATOR_MERGER_TIMEOUT_S   = 20.0


# ─── Local vision fallback ─────────────────────────────────────────────
# When the cloud Claude vision call fails or the Claude backend is off,
# retry against the local VLM served by the same Ollama instance.
# qwen2.5vl:7b uses ~8 GB VRAM with strong OCR + screenshot
# understanding, fits comfortably on the 3090 alongside the 8B text
# model (Ollama loads/unloads as needed). Set to "llava:13b" for
# slightly better natural-image describe at ~10 GB VRAM, or "off" to
# disable. Local-vision replies are prefixed `[local-vision] `.
LOCAL_VISION_FALLBACK = True
LOCAL_VISION_MODEL    = "qwen2.5vl:7b"


# ─── Local image generation (skills/image_gen.py) ──────────────────────
# Render text-to-image on the 3090 via ComfyUI's HTTP API or HF
# `diffusers`. Default 'off' so the skill registers cleanly but won't
# allocate ~6 GB of VRAM until flipped on. SDXL-Turbo finishes a
# 1024×1024 image in ~1-2 s on the 3090. Generated images land in
# ./screenshots/JARVIS_generated/ and auto-open in the default viewer.
IMAGE_GEN_BACKEND     = "off"            # 'comfyui' | 'diffusers' | 'off'
IMAGE_GEN_MODEL       = ""               # blank → backend-specific default
IMAGE_GEN_COMFYUI_URL = "http://localhost:8188"
IMAGE_GEN_STEPS       = 4                # SDXL-Turbo's sweet spot


# ─── TTS voice + legacy whisper ────────────────────────────────────────
TTS_VOICE     = "en-GB-RyanNeural"   # British male — closest to JARVIS (Paul Bettany)
WHISPER_MODEL = "base"               # legacy fallback name (tiny|base|small|medium|large)


# ─── TTS backend selector ──────────────────────────────────────────────
# synthesise() consults this on every utterance so 'use my voice' /
# 'switch to edge voice' takes effect immediately.
#   'edge'    → Microsoft Edge neural voice (current default, needs network)
#   'pyttsx3' → offline Windows SAPI / espeak (legacy fallback, robotic)
#   'xtts'    → local Coqui XTTS-v2 voice clone via skills/custom_voice.py
#                (~3 GB VRAM on the 3090; needs XTTS_VOICE_SAMPLE pointing
#                at a ~10 s WAV of the voice to clone). Falls back to edge
#                on any load / render error so a missed dep never silences
#                JARVIS.
TTS_BACKEND       = "edge"
XTTS_VOICE_SAMPLE = ""           # absolute path to a ~10 s WAV (mono, 24 kHz)
XTTS_LANGUAGE     = "en"         # ISO-639-1 hint for XTTS-v2


# ─── Voice pipeline selector ───────────────────────────────────────────
# Picks which speech loop drives the main UX.
#   'turn_based' → record_speech() → transcribe() → synthesise() → play
#                  (default; historical pipeline). Latency ~3-5 s.
#   'realtime'   → core.realtime_voice.RealtimeVoicePipeline streams
#                  partial transcripts in and synthesised audio out with
#                  single-queue barge-in. Drops perceived latency to
#                  <500 ms but requires the optional RealtimeSTT +
#                  RealtimeTTS deps. Falls back to 'turn_based' when
#                  those deps are missing — see is_available().
#
# Read in the hot path by core/voice_pipeline.realtime_enabled(); the monolith
# branches on it and ALWAYS falls back to turn_based on any error so an
# uninstalled optional dep never breaks the default loop. Override per-machine
# with the JARVIS_VOICE_MODE env var (it wins over this constant — see
# voice_pipeline._cfg).
VOICE_MODE = "turn_based"

# ─── Neural wake-word in standby (experimental) ────────────────────────
# When True, bobert_companion._handle_sleep_standby uses the neural detector in
# core/wake_word.py (openWakeWord / Porcupine) to spot the wake phrase in the
# captured audio buffer INSTEAD of running a full Whisper transcription of every
# overheard utterance just to substring-match WAKE_PHRASES. Default False keeps
# the historical Whisper-substring standby path byte-for-byte. On ANY detector
# error the monolith falls back to that Whisper path for the rest of the
# session. Read in the hot path by core/voice_pipeline.wake_word_autostart_-
# enabled(); override with the JARVIS_WAKE_WORD_AUTOSTART env var.
#
# NOTE: this is INDEPENDENT of skills/wake_listener.py's own WAKE_WORD_AUTOSTART
# constant, which governs that skill's separate always-on background detector
# (the one that nudges a sleeping main loop awake via proactive_announce). This
# flag only swaps the in-loop standby transcription strategy.
WAKE_WORD_AUTOSTART = False

# Alexa-style wake-word mode: when True, JARVIS BOOTS into wake-word standby —
# silent until you say "JARVIS", then it answers one turn and (in ambient mode)
# returns to standby — instead of always-listening. Seeds the sleep/standby
# latches at startup (a persisted crash-survival sleep state still wins). Pairs
# well with WAKE_WORD_AUTOSTART=True (neural detection vs Whisper-on-noise).
# Env override JARVIS_START_IN_STANDBY.
START_IN_STANDBY = False

# When True (default), JARVIS ignores spoken commands while SUSTAINED room music
# is playing UNLESS they start with the wake word "JARVIS" — so it doesn't reply
# to song lyrics it overhears. Set False if it keeps cutting YOU off while your
# own music plays; you can then talk to it normally over the music.
AMBIENT_MUSIC_REFUSE_WAKE = True


# ─── Whisper STT (faster-whisper preferred, GPU when present) ──────────
# `WHISPER_DEVICE = 'auto'` lets ctranslate2 + torch decide; 'cuda'
# forces GPU; 'cpu' forces the legacy path. large-v3-turbo on the 3090
# runs ~15× real-time at near-identical accuracy to large-v3.
WHISPER_DEVICE      = "auto"            # "auto" | "cuda" | "cpu"
WHISPER_MODEL_CUDA  = "large-v3-turbo"  # ~3.1 GB VRAM, 8x faster than large-v3
WHISPER_MODEL_CPU   = "small"           # CPU-friendly default when no GPU


# ─── Audio ducking (WASAPI session volume during JARVIS speech) ────────
# While JARVIS speaks, drop matching processes' WASAPI session volume to
# AUDIO_DUCKING_LEVEL, then fade back up on completion. Uses pycaw
# (Windows only); silently no-ops if pycaw isn't installed. AUDIO_-
# DUCKING_TARGETS lives near _duck_session() so the case-insensitive
# substring match stays close to the matching code.
AUDIO_DUCKING_ENABLED = True
AUDIO_DUCKING_LEVEL   = 0.25       # target scalar 0.0–1.0 (25% of current)
AUDIO_DUCKING_FADE_MS = 200        # fade duration each way


# ─── Mission narration (multi-action chain announcement) ──────────────
# When the LLM plans MISSION_NARRATION_THRESHOLD or more chained
# `[ACTION:]` tokens in one reply, speak an opening line and a one-line
# cue before each step. Suppresses the trailing prose so JARVIS doesn't
# double-speak.
MISSION_NARRATION_ENABLED   = True
MISSION_NARRATION_THRESHOLD = 3    # minimum action count to trigger narration


# ─── Mid-task status (anti-freeze single dry status line) ─────────────
# When a long-running action (auto-play streaming, upgrade pipeline,
# overnight ideas, dossier compile) hasn't returned by MID_TASK_-
# STATUS_DELAY seconds, speak a single dry status line so JARVIS
# doesn't feel frozen. Allow-list lives in LONG_RUNNING_ACTIONS, phrase
# bank in _MID_TASK_STATUS_LINES.
MID_TASK_STATUS_ENABLED = True
MID_TASK_STATUS_DELAY   = 8.0      # seconds before the dry status line fires


# ─── Audio capture (VAD + sample rate) ─────────────────────────────────
# 2026-05-30 [self-heal]: lowered 0.010 → 0.008 so VAD still trips after
# AEC's duck gain (now 0.7) shaves the input by ~30% during JARVIS's own
# playback. Was previously catching live speech post-AEC-attn at 0.007 and
# falling under the 0.010 floor, producing the "VAD never tripped" stall.
VAD_THRESHOLD = 0.008              # mic RMS for speech; raise to ignore noise
SILENCE_SECS  = 1.4                # seconds of quiet before processing
SAMPLE_RATE   = 16000              # mic capture sample rate (Hz)


# ─── Audio processor tuning (AEC fallback + AGC flatness gate) ─────────
# 2026-05-30 [self-heal]: extracted from core/audio_processor.py defaults
# so they can be tuned without editing the module. The diagnostic that
# raised this fix flagged the prior 0.4 duck-gain as the most likely
# culprit — ducking 60% of the mic on every TTS spillover knocked normal
# speech below VAD_THRESHOLD whenever JARVIS had recently spoken.
#
# AEC_DUCK_GAIN — fallback echo-suppression gain applied when JARVIS
# played audio within the last 150 ms and the WebRTC APM isn't available.
# 0.7 = 30% attenuation (preserves user speech); 0.4 was too aggressive.
AEC_DUCK_GAIN = 0.7

# AGC_FLATNESS_{MIN,MAX} — bounds on the AGC's spectral-flatness gate.
# The smoothed flatness drifts under long silence-then-noise patterns and
# can eventually pin the gate closed (no gain applied → VAD never trips).
# Clamping keeps it inside the sigmoid's usable band so gain recovers.
AGC_FLATNESS_MIN = 0.20
AGC_FLATNESS_MAX = 0.80

# MIC_SILENT_WARN_SECONDS — how long record_speech can poll chunks where
# the raw mic RMS is effectively zero before emitting a one-time silent-
# mic warning. Distinguishes "user is silent" from "mic is hardware-dead".
MIC_SILENT_WARN_SECONDS = 30.0


# ─── Cameras (multi-camera attention tracking) ─────────────────────────
# "primary" camera tracks face position precisely (eyes follow you).
# Side cameras only fire when ONLY they see the face — the robot looks
# toward that camera's direction, so when you turn to a different
# monitor the robot's eyes turn with you. look_x / look_y are where
# the robot should aim its eyes (0.0–1.0) when this camera is the only
# one that can see your face. Ignored for primary.
CAMERAS = [
    {"index": 1, "label": "Left webcam (left monitor)",          "primary": False, "look_x": 0.15, "look_y": 0.5},
    {"index": 0, "label": "Right webcam (top of right monitor)", "primary": True,  "look_x": 0.85, "look_y": 0.5},
]

# Camera probe — if CAMERAS fails to open, sweep indices 0..MAX-1 and
# rewrite CAMERAS with whatever's actually plugged in. cv2.CAP_DSHOW
# retries internally for ~26s on a missing index, so each probe is
# capped by CAMERA_PROBE_TIMEOUT_SEC.
CAMERA_PROBE_ENABLED      = True
CAMERA_PROBE_MAX          = 12     # probe indices 0..MAX-1
CAMERA_PROBE_TIMEOUT_SEC  = 3.0    # per-index hard timeout

# Processes that commonly hold exclusive locks on webcams. If the probe
# finds zero cameras, we scan for these and surface them so the user
# knows what to close.
CAMERA_LOCK_PROCESSES = {
    "teams.exe", "ms-teams.exe", "msteams.exe",
    "zoom.exe", "cpthost.exe",
    "obs64.exe", "obs32.exe", "obs.exe",
    "skype.exe", "skypeapp.exe",
    "discord.exe", "discordcanary.exe", "discordptb.exe",
    "webex.exe", "webexmta.exe", "atmgr.exe",
    "slack.exe",
    "googlemeet.exe", "meet.exe",
    "manycam.exe", "snapcamera.exe", "facerig.exe", "vmix.exe",
    "logi capture.exe", "logitune.exe", "logioptionsplus.exe",
    "windowscamera.exe", "cameraapp.exe",
    "nvbroadcast.exe", "nvidia broadcast.exe",
}


# ─── Monitor layout ────────────────────────────────────────────────────
# Friendly names for each monitor. Each entry is (x, y, w, h). JARVIS
# uses this for "open Google on my left monitor"-style requests and for
# gaze direction reporting.
MONITORS = {
    "left":   (-2560, 0,     2560, 1440),
    "middle": (0,     0,     2560, 1440),
    "right":  (2560,  0,     2560, 1440),
    "top":    (0,     -1440, 2560, 1440),
}

# Which monitor to move the JARVIS console window to at startup.
# Must match a key in MONITORS above; "" / None = leave wherever it is.
CONSOLE_MONITOR = "top"


# ─── On-screen overlays + tray ─────────────────────────────────────────
# THE HUD. As of 2026-05-30 the single unified HUD (hud/jarvis_unified_hud.py)
# is the one and only on-screen status surface — draggable, resizable, and
# remembers its position/size. It REPLACES the old sprawl of overlapping
# overlays (workshop HUD, workshop canvas, workshop print monitor, bambu
# corner overlay, arc-reactor status ring, holographic fullscreen + holo HUD
# v2, briefing card). All of those are retired below by turning their
# auto-launch flags OFF. User: "too many huds … i want a fully upgraded one
# fully feature packed."
HUD_ENABLED = True                 # drives the unified HUD at boot
HUD_MONITOR = "top"                # which monitor in MONITORS to anchor to

# Full-virtual-screen translucent target reticle that flashes for ~2s wherever
# JARVIS performs a UI-automation action. KEPT ON — it is click-feedback, not
# an info widget, so it isn't part of the HUD clutter and is invisible except
# during the brief flash. It is click-through and never needs repositioning.
RETICLE_OVERLAY_ENABLED = True

# System-tray applet (tray.py at project root, pystray + Pillow).
TRAY_ENABLED = True

# ── Retired overlays (all superseded by the unified HUD) ────────────────
# Each of these used to auto-spawn its own frameless, non-movable widget.
# Their data (system vitals, JARVIS state reactor, Bambu print progress) now
# lives in the unified HUD, so they are all forced OFF. Flip any back to True
# only if you specifically want that standalone surface again.
HOLOGRAPHIC_OVERLAY_AUTO_LAUNCH   = False   # fullscreen holo overlay
HOLO_WORKSHOP_AUTO_ON_THINK       = False   # compact rotating arc-reactor canvas
WORKSHOP_HUD_AUTO_LAUNCH          = False   # top-right CPU/RAM/bambu widget
WORKSHOP_PRINT_MONITOR_AUTO_LAUNCH = False  # top-center Stark print panel
BAMBU_OVERLAY_AUTO_WHILE_PRINTING = False   # top-right bambu corner overlay


# ─── Bambu H2D 3D printer credentials ──────────────────────────────────
# Leave blank to disable monitoring. Pull from the printer's touchscreen
# → LAN Only (IP + access code) and Bambu Handy → Firmware Version (SN).
BAMBU_PRINTER_IP  = os.getenv("BAMBU_PRINTER_IP",  "")   # env/.env only - never commit
BAMBU_ACCESS_CODE = os.getenv("BAMBU_ACCESS_CODE", "")   # env/.env only - never commit
BAMBU_SERIAL      = os.getenv("BAMBU_SERIAL",      "")   # env/.env only - never commit


# ─── iTunes auto-launch (Apple Music COM bridge) ───────────────────────
# When True, `_get_itunes()` will spawn iTunes.exe if the COM Dispatch
# can't find a running instance. When False (default), the music
# actions return a friendly error and do NOT pop iTunes open. Apple
# Music / Spotify / YouTube streaming actions are unaffected — those
# use the browser auto-play pipeline, not iTunes COM. NOTE: the parent
# module still calls _itunes_bridge.set_auto_launch(ITUNES_AUTO_LAUNCH)
# right after the import so the bridge picks the live value at boot.
ITUNES_AUTO_LAUNCH = False


# ─── Overnight self-improvement engine ─────────────────────────────────
# OVERNIGHT_UPGRADE_ENABLED = True means the background thread polls
# for idle + gap thresholds and fires the upgrade pipeline on its own.
# Flip to False to disable the auto-fire — manual `upgrade` actions are
# also gated against this flag (queued task 2026-05-29 11:25).
OVERNIGHT_UPGRADE_ENABLED  = False  # PAUSED 2026-05-30 pending post-fix stabilization (was True)
OVERNIGHT_IDLE_MINUTES     = 30    # minutes of silence before a cycle fires
OVERNIGHT_CYCLE_GAP_HOURS  = 0.5   # minimum hours between cycles
OVERNIGHT_MODE_HOURS       = 8     # how long .overnight_active persists

# ─── Update checker ────────────────────────────────────────────────────
# UPDATE_CHECK_ENABLED = True lets a running instance compare itself to the
# latest GitHub release on boot (once/day, cached in data/update_check.json)
# and queue a single spoken nudge when a newer version exists. Needs a token
# (JARVIS_GITHUB_TOKEN / GITHUB_TOKEN) for the private repo; degrades silently
# without one. See core/update_checker.py.
UPDATE_CHECK_ENABLED       = True


# ─── Personal-files RAG (Khoj-style second brain) ──────────────────────
# Watched folders, embed, semantic search. Read by skills/personal_rag.py
# at autostart. Every knob can be changed at runtime via the
# rag_configure action ('paths=…', 'embed_model=…', etc).
# RAG_EMBED_MODEL — Ollama embedding model. 'nomic-embed-text' runs on
# the 3090 at 200+ docs/sec.
RAG_INDEX_PATHS = [
    os.path.join(os.path.expanduser("~"), "Documents"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.join(os.path.expanduser("~"), "OneDrive"),
]
RAG_EMBED_MODEL     = "nomic-embed-text"
RAG_OLLAMA_ENDPOINT = "http://localhost:11434/api/embeddings"
RAG_RERANKER_MODEL  = "BAAI/bge-reranker-base"


# ─── Robot eye / mouth scaling (Phase 1E) ──────────────────────────────
# Robot-only tuning. Inert when ROBOT_ENABLED=False; kept here so
# core/state.py doesn't need a back-reference into bobert_companion for
# the audio_master + debug flag seeds below.
MIRROR_EYES_X = False
MIRROR_EYES_Y = False
MOUTH_SCALE   = 9.0     # RMS → mouth-open amount; raise if mouth barely moves


# ─── Audio processor (Phase 1E) ────────────────────────────────────────
# Real-time mic cleanup (core/audio_processor.py): echo-cancel JARVIS's
# own playback, suppress stationary background noise, AGC to a target
# RMS before STT. VAD still runs on the RAW signal so existing
# VAD_THRESHOLD tuning isn't invalidated — the processed chunks are
# what we feed to Whisper.
AUDIO_PROCESSING_ENABLED = True

# Per-stage switches under the AUDIO_PROCESSING_ENABLED master. Each seeds
# the matching runtime toggle in core/state.py (_audio_aec/ns/agc_enabled),
# which the tray's Audio Controls submenu flips at runtime. Default True =
# the prior hardcoded behaviour (all three stages on), so nothing changes
# unless the user turns one off in the Settings GUI / user_settings.json.
#   AUDIO_ECHO_CANCEL    — cancel JARVIS's own playback bleeding into the mic
#   AUDIO_NOISE_SUPPRESS — suppress stationary background noise
#   AUDIO_AGC            — auto-gain the mic to a target RMS before STT
AUDIO_ECHO_CANCEL    = True
AUDIO_NOISE_SUPPRESS = True
AUDIO_AGC            = True


# ─── VAD debug print (Phase 1E) ────────────────────────────────────────
# When True, "[vad] peak RMS = X" prints after each utterance so you can
# tune VAD_THRESHOLD precisely. Flipped at runtime via the tray Debug
# Mode toggle, which mirrors into _debug_mode[0] in core/state.py.
VAD_DEBUG = True


# ─── Standby auto-engage loop (skills/standby_audio_detect) ────────────
# Independent background loop that runs whisper-tiny on a short rolling
# mic buffer to catch *vocal* music (intelligible lyrics) the spectral
# classifier alone can miss. When sustained-lyric content is detected
# for STANDBY_LOOP_MATCH_WINDOWS consecutive checks AND the headset is
# the active output, JARVIS auto-engages standby/wake-word-only mode and
# TTS "I'll wait until you call, sir". Off entirely without librosa.
STANDBY_LOOP_ENABLED              = True
STANDBY_LOOP_BUFFER_SECONDS       = 3.0     # rolling mic buffer fed to whisper
STANDBY_LOOP_CHECK_INTERVAL_SEC   = 5.0     # cadence between checks
STANDBY_LOOP_MATCH_WINDOWS        = 3       # consecutive windows to trip (3 × 5s = 15s)
STANDBY_LOOP_ONSET_ENERGY_MIN     = 0.30    # librosa onset_strength mean ≥ this = musical
STANDBY_LOOP_RHYME_RATIO_MIN      = 0.30    # share of word-pairs sharing 2-char suffix
STANDBY_LOOP_WHISPER_MODEL        = "tiny"  # whisper model name (kept small for latency)


# ─── User settings overrides (data/user_settings.json) ─────────────────
# The tray Settings GUI (tools/settings_window.py) writes data/user_settings.json
# (gitignored). Apply those overrides over the defaults above so a saved setting
# takes effect on the next start — and because bobert_companion.py does
# `from core.config import *` AFTER this module finishes importing, every
# consumer (the monolith, core.voice_pipeline, the skills) sees the overridden
# value with no extra wiring.
#
# Safe + best-effort (the second import-time I/O in this file, after
# RAG_INDEX_PATHS): we override ONLY a constant that already exists here — so the
# GUI's schema, which is curated FROM this file, is the allow-list — coerce to
# the existing constant's type, and leave the default on any error. A missing
# file (fresh install, before the GUI ever ran) is a silent no-op.
def _apply_user_settings() -> None:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "user_settings.json")
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    g = globals()
    for key, val in data.items():
        if key.startswith("_") or key not in g:
            continue          # only override an existing public config constant
        cur = g[key]
        try:
            if isinstance(cur, bool):
                g[key] = bool(val)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                g[key] = int(val)
            elif isinstance(cur, float):
                g[key] = float(val)
            elif isinstance(cur, str):
                g[key] = str(val)
            elif isinstance(cur, (list, tuple)) and isinstance(val, (list, tuple)):
                g[key] = type(cur)(val)
            elif isinstance(cur, dict) and isinstance(val, dict):
                g[key] = {**cur, **val}   # merge so a PARTIAL override keeps the other keys
            # other / mismatched types: keep the default rather than risk a bad value
        except Exception:
            continue


_apply_user_settings()
