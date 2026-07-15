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
SCREEN_VISION_MODEL   = "claude-sonnet-5"


# ─── UI automation ─────────────────────────────────────────────────────
# Click, type, navigate apps autonomously (needs: pip install pyautogui pillow).
UI_AUTOMATION_ENABLED = True


# ─── Skills system ─────────────────────────────────────────────────────
# JARVIS can write new Python modules under skills/ to teach himself
# new tasks. Loaded at startup by load_skills().
SKILLS_ENABLED = True

# ─── Tiered long-term memory ───────────────────────────────────────────
# Wires core/long_term_memory.py into the conversation loop: every turn is
# recorded on a background thread and the top-k relevant semantic facts are
# injected into the volatile system-prompt tail (budget-bounded, never
# blocks the voice loop). Overridable via data/user_settings.json.
LTM_ENABLED = True
# Torch device for the LTM sentence-transformer embedder (bge-small, ~0.4GB).
# "cpu" (default 2026-07-15): keep the embedder OFF the GPU — encoding a short
# utterance on the 14900K is a few ms (well inside the 0.6s recall budget), and
# the freed VRAM goes to the local LLM (now the 26B), the tenant that actually
# needs it. The old "" auto-default silently loaded it on CUDA on this box. Set
# "" to restore auto (cuda if available), or a specific device to override.
LTM_EMBED_DEVICE = "cpu"

# ─── Streaming TTS (sentence-flush) ────────────────────────────────────
# Speak the first complete, action-free sentence(s) of a Claude reply WHILE
# the rest is still streaming, instead of waiting for the full completion —
# the perceived-latency win core/llm_client.stream_text() was built for.
# Conservative by design (see bobert_companion._SentenceFlushBuffer): early
# speech hard-stops at the first '[' (possible [ACTION:]/tag marker), is
# capped at 2 sentences, and the downstream speaker skips whatever was
# already voiced. When False, the Claude turn uses the blocking complete()
# exactly as before. Overridable via data/user_settings.json.
STREAMING_TTS_ENABLED = True

# ─── Streaming auto-fullscreen ─────────────────────────────────────────
# After JARVIS starts a TV show / movie stream and playback is CONFIRMED
# started, send the player fullscreen ('f' works on YouTube / Netflix /
# Disney+ / Prime / Hulu / Max) so the show fills the screen without the user
# reaching for the keyboard. The per-service key lives in the streaming config
# (bobert_companion._STREAMING_SERVICES[…]["fullscreen_key"], default 'f',
# None disables for that one service); THIS flag is the master switch across
# every service. When False, JARVIS starts playback and leaves the player
# windowed. Read live at play time (mirrors the STREAMING_TTS_ENABLED fresh-
# import pattern in bobert_companion._streaming_go_fullscreen) so the current
# value is honoured at each play. 2026-07-08: corrected an overpromise — a
# Settings-GUI / user_settings.json flip takes effect on the NEXT start, not
# live: _apply_user_settings() runs once at import, and a fresh `import
# core.config` returns the already-cached module without re-reading the file.
# Overridable via data/user_settings.json.
STREAMING_AUTO_FULLSCREEN = True

# ─── Barge-in (wake-word interrupt during TTS) ─────────────────────────
# When True, a wake-word ENGINE hit (openwakeword/porcupine via
# skills/wake_listener.py — NOT loose transcript matching) that lands while
# JARVIS is actively speaking cuts TTS playback immediately so the user never
# has to wait out a long reply. The wake announcement is swallowed on a
# barge-in: JARVIS simply goes quiet and listens. Echo-safety lives in
# bobert_companion.request_tts_interrupt(): if the sentence currently being
# spoken contains "jarvis" the interrupt is refused, so the speakers saying
# his own name can never self-interrupt through the mic.
#
# NOTE: this is intentionally a SEPARATE knob from the legacy module-level
# BARGE_IN_ENABLED constant inside bobert_companion.py (the RMS/headset
# InputStream path, hard-disabled there after the 0xc0000374 PortAudio
# use-after-free). This knob only gates the new wake-word interrupt path,
# which opens NO extra stream — the wake listener already owns its own mic.
# Read live via `core.config` at interrupt time (mirrors the
# STREAMING_TTS_ENABLED fresh-import pattern) so the current value is honoured
# at each interrupt. 2026-07-08: corrected an overpromise — a Settings-GUI /
# user_settings.json flip takes effect on the NEXT start, not live:
# _apply_user_settings() runs once at import and a fresh `import core.config`
# returns the already-cached module without re-reading the file. When False the
# behaviour is byte-identical to pre-barge-in builds.
# Overridable via data/user_settings.json.
BARGE_IN_ENABLED = True


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
# claude-sonnet-5: near-Opus intelligence at the same list price Sonnet 4.6 had
# ($3/$15 per MTok) — the best default brain per dollar as of 2026-07. Users who
# want the ceiling can pick claude-opus-4-8 ($5/$25) in the Settings GUI.
CLAUDE_MODEL = "claude-sonnet-5"
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
# and the experience is meant to be good on its own. The baseline brain is
# gemma4:26b-a4b-it-qat (2026-07): a 26B MoE with 4B ACTIVE params, so it
# generates at small-model speed with big-model quality, is MULTIMODAL
# (text + image — the same resident model can serve local vision, no second
# VLM co-load), and its 16 GB QAT quant leaves real headroom next to whisper
# on a 24 GB card. The call path still picks num_ctx per-model (see
# _local_num_ctx: 30B-class tags get 12k, everything else 16k). For local
# calls the giant Claude-tuned PC_CONTROL_PROMPT is swapped for a compact
# action cheatsheet (see _local_cheatsheet). The runtime selector
# `_get_local_llm_model()` consults JARVIS_LOCAL_LLM_MODEL first, then walks
# a fallback chain (gemma4:26b-a4b → qwen3:30b-a3b → qwen2.5:14b →
# llama3.1:8b → first available tag). The old dense qwen2.5:32b default
# (~22 GB resident) was retired — it left no headroom and bricked the GPU
# whenever vision or whisper co-loaded.
LOCAL_LLM_FALLBACK = True
# 2026-07-15 (P2 phase B): PROMOTED to gemma4:26b-a4b-it-qat now that TTS moved
# OFF the 3090 (Kokoro-on-CPU) freed ~13GB. The old "26b-a4b returns EMPTY output
# (ollama #15428/#16456)" claim is STALE — a fresh on-box measurement proved it
# fixed on the current ollama: 0 empties across 5 real >7k-token-prompt turns,
# 94-110 tok/s, and 18794MiB resident with the brain alone → ~5.8GB free (vs the
# old ~1.8GB beside the resident clone). It's a 26B MoE (4B active) so it's both
# smarter and fast, and — critically — MULTIMODAL, so chat+vision keep sharing ONE
# resident model (no chat<->VLM swap thrash). Thinking disabled via think:false
# (_local_think_param: gemma4* → false). gemma4:12b is retained as the graceful
# lower-VRAM fallback (_LOCAL_LLM_PREFERENCE[1] + the empty-response failover).
# The on-demand voice clone is now VRAM-GATED (core/voice_clone._resolve_device):
# 26B leaves 5.8GB free < the clone's ~6GB gate, so arming the clone degrades to
# Kokoro instead of OOM-contending the card. qwen3:30b-a3b stays a text-only opt-in
# "max brain" (breaks the shared-vision property). Override via JARVIS_LOCAL_LLM_MODEL.
LOCAL_LLM_MODEL    = "gemma4:26b-a4b-it-qat"
# 127.0.0.1, NEVER "localhost" (2026-07-12): Windows resolves localhost to
# ::1 first, and when Ollama listens only on IPv4 the ::1 attempt eats a
# measured, rock-steady ~2.05s before falling back — pushing every request
# just past _ollama_alive()'s 2s probe. Result: Ollama up and healthy while
# EVERY availability check said dead ("local vision unavailable"), and the
# self-heal kept restarting a server that was never down. Same pin applied
# to every other 11434 reference in the tree.
LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434"

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

# CHAPPIE_ENABLED — autostart the continuous self-learning daemon
#   (skills/chappie_consciousness.py). False by default because the daemon
#   spends Claude budget (up to DAILY_BUDGET_USD/day) the moment it runs.
#   The skill's recall actions register regardless; only the spending
#   background thread is gated by this flag. Flip via the Settings GUI /
#   user_settings.json to opt in.
CHAPPIE_ENABLED = False

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
ORCHESTRATOR_PLANNER_MODEL      = "claude-sonnet-5"
ORCHESTRATOR_WORKER_MODEL       = "claude-haiku-4-5"
ORCHESTRATOR_MERGER_MODEL       = "claude-sonnet-5"
ORCHESTRATOR_MAX_PARALLEL       = 4
ORCHESTRATOR_WORKER_TIMEOUT_S   = 30.0
ORCHESTRATOR_PLANNER_TIMEOUT_S  = 20.0
ORCHESTRATOR_MERGER_TIMEOUT_S   = 20.0


# ─── Local vision fallback ─────────────────────────────────────────────
# When the cloud Claude vision call fails or the Claude backend is off,
# retry against the local VLM served by the same Ollama instance.
# Local-vision replies are prefixed `[local-vision] `.
#
# 2026-07: the default LOCAL_VISION_MODEL is now the SAME multimodal tag as
# the chat baseline (gemma4:26b-a4b-it-qat handles text + images). When the
# vision model equals the resident chat model, a vision fallback re-uses it
# — no second model is loaded and the historical over-commit can't happen.
# The flag still ships False (2026-06-06 rationale below) so a box whose
# user PINNED a separate VLM (e.g. qwen2.5vl:7b next to a ~21 GB dense chat
# model) can't brick on a transient Claude APIStatusError / network blip:
# the fallback fires on those too, not just explicit local requests. Flip
# True (Settings GUI / user_settings.json) when the vision tag matches the
# chat tag, or when there is VRAM headroom for both models at once.
# Set LOCAL_VISION_MODEL to "off" to disable local vision entirely.
LOCAL_VISION_FALLBACK = False
# Same tag as LOCAL_LLM_MODEL: gemma4:26b-a4b is multimodal, so vision re-uses the
# RESIDENT chat model — no eviction/swap, no over-commit (the chat<->VLM swap
# was wedging llama-server mid-eviction, live outage 2026-07-10 09:13). Kept in
# lockstep with LOCAL_LLM_MODEL so promoting the brain never forks vision onto a
# second VLM. Vision verified on-box with think:false.
LOCAL_VISION_MODEL    = "gemma4:26b-a4b-it-qat"


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


# ─── Local voice-cloning backend (Chatterbox) ──────────────────────────
# A SEPARATE, opt-in path from the XTTS backend above: Resemble AI's
# Chatterbox (MIT) clones a voice from a ~5 s consented reference clip and
# renders on the RTX 3090. core.voice_clone.is_available() gates it and it
# ALWAYS falls back to the edge-tts → pyttsx3 → SAPI5 ladder on any failure —
# a missing dep / no-GPU box / unselected profile never silences JARVIS.
#
# ETHICS: only the owner's OWN consented voice or a JARVIS in-character
# (non-celebrity) voice — enrollment requires an explicit consent flag and
# profiles/audio live under a gitignored dir (never committed). See
# core/voice_clone.py's module docstring.
#
# Read live by synthesise() every utterance so a 'switch to my voice' voice
# action / a user_settings.json flip takes effect immediately. Default OFF.
VOICE_CLONE_ENABLED = False      # master switch (OFF by default)
VOICE_CLONE_PROFILE = ""         # active profile name under data/voice_profiles/
VOICE_CLONE_MODEL   = "chatterbox"   # engine id (currently only "chatterbox")
# Torch device for the clone engine. "" = historical default (cuda:0 if present
# else cpu). Set "cuda:1" to run chatterbox on a SECOND, idle GPU so it stops
# eating the primary card's VRAM (frees ~3GB on the 3090 for the LLM). Gated by
# a free-VRAM check in core/voice_clone (degrades to the edge-tts ladder if the
# chosen device lacks room), so a bad value never OOMs. 2026-07-09.
VOICE_CLONE_DEVICE  = ""


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

# Manual "wake-word mode" (Alexa-style): when True, JARVIS ignores EVERY spoken
# command that doesn't start with the wake word "JARVIS" — one utterance, one
# command. Off by default; the user flips it on by voice (e.g. for an external
# TV the OS media session can't see). _apply_user_settings() overrides this from
# user_settings.json, and the voice action toggles it live at runtime.
REQUIRE_WAKE_MODE = False


# ─── Whisper STT (faster-whisper preferred, GPU when present) ──────────
# `WHISPER_DEVICE = 'auto'` lets ctranslate2 + torch decide; 'cuda'
# forces GPU 0; 'cuda:N' pins STT to a specific GPU (e.g. 'cuda:1' to run
# Whisper on a second card and keep the primary free for the LLM/voice);
# 'cpu' forces the legacy path. large-v3-turbo on the 3090 runs ~15× real-time
# at near-identical accuracy to large-v3.
WHISPER_DEVICE      = "auto"            # "auto" | "cuda" | "cuda:N" | "cpu"
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


# ─── Focus mode / do-not-disturb (skills/focus_mode.py) ────────────────
# FOCUS_MODE_ENABLED — makes the do-not-disturb "focus mode" FEATURE available
#   (the voice actions focus_mode_on / focus_mode_off / whats_missed and the
#   proactive_announce gate that holds unsolicited announcements while focused).
#   The FEATURE being available does NOT mean the mode is engaged: focus mode
#   always starts OFF at boot and is only turned on by an explicit command
#   ("focus mode on", "do not disturb", "quiet mode"). This knob is a global
#   kill-switch — set False and the gate in bobert_companion.proactive_announce
#   short-circuits to a no-op (announcements are never held) so a bad focus
#   state can never silence JARVIS. When focus mode is active, ONLY unsolicited
#   proactive speech is held; wake-word + direct command responses (which call
#   _speak, not proactive_announce) are never affected. Overridable via
#   data/user_settings.json like every other flag.
FOCUS_MODE_ENABLED = True


# ─── Audio capture (VAD + sample rate) ─────────────────────────────────
# 2026-05-30 [self-heal]: lowered 0.010 → 0.008 so VAD still trips after
# AEC's duck gain (now 0.7) shaves the input by ~30% during JARVIS's own
# playback. Was previously catching live speech post-AEC-attn at 0.007 and
# falling under the 0.010 floor, producing the "VAD never tripped" stall.
VAD_THRESHOLD = 0.008              # mic RMS for speech; raise to ignore noise
SILENCE_SECS  = 1.4                # seconds of quiet before processing
SAMPLE_RATE   = 16000              # mic capture sample rate (Hz)


# ─── Capture auto-gain (quiet-mic normalization before Whisper) ────────
# CONSERVATIVE input normalization applied to the recorded float32 buffer
# right BEFORE faster-whisper sees it, on BOTH the normal turn and the
# standby/wake path. A quiet mic records speech at a low peak RMS
# (~0.01–0.06) where Whisper returns an EMPTY string, so the wake word
# "JARVIS" is never heard. This boosts such audio toward a usable level
# WITHOUT touching already-good audio.
#
# The helper apply_capture_auto_gain() is a pure no-op unless the captured
# peak RMS sits in the band (NOISE_FLOOR, TARGET_PEAK):
#   • peak ≥ TARGET_PEAK            → already loud enough, gain 1.0 (untouched)
#   • peak ≤ NOISE_FLOOR            → pure silence/room hiss, gain 1.0 (so we
#                                     never amplify noise into Whisper
#                                     hallucinations)
#   • NOISE_FLOOR < peak < TARGET   → gain = min(MAX, TARGET/peak), hard-clipped
#                                     to [-1, 1] to prevent overflow distortion.
# Read live via `core.config` so the Settings GUI / user_settings.json
# override path reaches them.
CAPTURE_AUTO_GAIN_ENABLED     = True
CAPTURE_AUTO_GAIN_TARGET_PEAK = 0.25   # boost quiet audio up toward this peak
CAPTURE_AUTO_GAIN_MAX         = 10.0   # never multiply by more than this
CAPTURE_AUTO_GAIN_NOISE_FLOOR = 0.005  # peak ≤ this = silence; never amplify


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
#
# "name" (OPTIONAL) — a case-insensitive substring of the DirectShow device
# friendly name (as `python bobert_companion.py --list-cameras` prints it).
# When present, _open_capture resolves the LIVE index by that name at open time
# (via pygrabber) and PREFERS it over the static "index" — so a USB
# re-enumeration that shuffles the indices (the mic-shuffle bug class) can't
# silently point the face tracker at the WRONG camera. The static "index" is the
# FALLBACK used only when the name doesn't resolve (pygrabber missing / device
# unplugged). Omit "name" to keep the historical pure-index behaviour. Set it to
# the owner's two webcams so a re-plug keeps tracking the right one.
CAMERAS = [
    # 2026-07-13: the LEFT webcam is now an eMeet C960 — the owner swapped out
    # the Logi C270 (it had been dropping off the USB bus; the C960 replaced it
    # while chasing camera flicker). Kinect sits under the centre monitor and is
    # handled separately (KINECT_AS_CAMERA/presence; never grab index 1, that's
    # the Kinect colour stream). VERIFIED live DirectShow order (pygrabber):
    # 0=USB 2.0 Camera, 1=Kinect V2 Video Sensor, 2=HD Webcam eMeet C960. The
    # "name" drives live resolution (USB replugs re-shuffle indices); these
    # indices are only the fallback.
    {"index": 2, "label": "Left webcam (left monitor)",          "name": "emeet c960",    "primary": True,  "look_x": 0.5,  "look_y": 0.5},
    {"index": 0, "label": "Right webcam (top of right monitor)", "name": "usb 2.0 camera", "primary": False, "look_x": 0.85, "look_y": 0.5},
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


# ─── Xbox Kinect v2 sensor (opt-in; all default False) ─────────────────
# The Kinect v2 adds true skeleton-based room presence, head-position gaze,
# a 1080p color camera, plus depth + infrared (night-vision) streams. It is
# OFF by default — it's a camera + microphone array pointed at the room, so
# every Kinect capability is opt-in and privacy-conscious. The bridge
# (audio/kinect_bridge.py) never opens the sensor unless KINECT_ENABLED is
# True. Flip these here (or via the matching JARVIS_* env override) to use it.
#
# KINECT_ENABLED — master switch. When False the bridge short-circuits every
#   accessor and never touches pykinect2 / the Kinect Runtime. Set True to let
#   the bridge open the sensor (Color | Body | Depth | Infrared).
KINECT_ENABLED = False
# KINECT_AS_CAMERA — when True, the face-tracking loop uses the Kinect's 1080p
#   color stream as a face-tracking camera (a KinectCapture stands in for
#   cv2.VideoCapture). Leave False to keep using the configured USB webcams; an
#   explicit CAMERAS entry with {"type": "kinect"} also opts a slot in.
KINECT_AS_CAMERA = False
# KINECT_PRESENCE_ENABLED — when True, the face-tracker skill merges real
#   skeleton presence (body count + head-facing) from the Kinect into its
#   gaze/presence state, beating the Haar-cascade guesswork when the sensor
#   can see the room.
KINECT_PRESENCE_ENABLED = False
# KINECT_PRESENCE_STANDBY — when True (and presence is enabled), JARVIS drops
#   to standby after the room has been empty for a sustained window. Off by
#   default so the sensor never silences JARVIS unless you ask it to.
KINECT_PRESENCE_STANDBY = False
# KINECT_PRESENCE_WAKE — when True (and presence is enabled), JARVIS clears
#   standby the moment a person reappears in the Kinect's view. Off by default.
KINECT_PRESENCE_WAKE = False
# KINECT_GAZE_ENABLED — when True, the Kinect becomes the PRIMARY "which monitor
#   am I looking at" signal: the face-tracker skill reads the nearest body's
#   head/shoulder facing YAW from the Kinect (audio.kinect_bridge.get_head_yaw)
#   and maps it to a monitor via the MONITORS layout, so which-monitor works with
#   BOTH WEBCAMS OFF. The legacy two-webcam look_x heuristic stays as a graceful
#   FALLBACK for when the Kinect has no body in view. Independent of
#   KINECT_PRESENCE_ENABLED (you can have gaze without the standby/wake
#   automations), but like every Kinect feature it needs KINECT_ENABLED so the
#   bridge actually opens the sensor. A sensible built-in yaw→monitor mapping
#   ships by default; per-desk tuning is optional via the 'calibrate gaze' voice
#   action (look at each monitor in turn), persisted to a SEPARATE gitignored
#   data/kinect_gaze_calibration.json — never user_settings.json. Off by default.
KINECT_GAZE_ENABLED = False
# KINECT_GESTURES_ENABLED — when True, a background poller reads the Kinect
#   skeleton stream (~18 Hz) and maps discrete gestures to actions: WAVE wakes
#   JARVIS from standby, RAISE_HAND confirms a pending confirmation (like saying
#   "yes"), and a SWIPE dismisses/cancels (stop speech + clear the pending
#   confirmation). Off by default; never runs in staging/test. See
#   audio/kinect_gestures.py (recognizer) + skills/kinect_gestures.py (wiring).
KINECT_GESTURES_ENABLED = False
# KINECT_POINT_CONTROL_ENABLED — when True, "point-to-control": the owner points
#   an arm at a real device (a desk lamp, a fan) and says "turn that on/off" and
#   JARVIS controls the right smart-home device. First calibrate each device
#   ("calibrate pointing for the desk lamp" while pointing at it) — the pointing
#   DIRECTION is stored in a separate gitignored data/kinect_pointing.json
#   (never user_settings.json) bound to the real device; then a pointed "turn
#   that on" resolves the live arm ray to the closest calibrated target within
#   ~18° and fires the EXISTING smart-home on/off path. Off by default; never
#   drives a device in staging/test. See audio/kinect_pointing.py (geometry +
#   store) + skills/kinect_pointing.py (wiring).
KINECT_POINT_CONTROL_ENABLED = False
# KINECT_AIR_MOUSE_ENABLED — when True, "air-mouse": point an OPEN hand at the
#   screen to move the cursor, CLOSE the hand to RIGHT-click, and hold it closed
#   to drag (close→open quickly = a right-click; close→move→open = a right-drag).
#   A background poller (~30 Hz) maps the pointing hand's position within a
#   calibrated reach-box onto the PRIMARY monitor, heavily EMA-smoothed to fight
#   jitter, and drives the cursor via win32api (pyautogui fallback). A glowing
#   JARVIS reticle (hud/jarvis_air_cursor.py) follows the cursor — cyan while
#   tracking an open hand, gold-locked on grab/drag. Off by default; never runs
#   in staging/test; a dead-man releases any held button the instant the hand
#   isn't tracked. See audio/kinect_bridge.get_hand_states() (grip) +
#   skills/kinect_air_mouse.py (wiring) + hud/jarvis_air_cursor.py (overlay).
KINECT_AIR_MOUSE_ENABLED = False
# ─── AIR-MOUSE SMART-ENGAGE knobs (2026-07, feat/smart-engage) ───────────────
#   The owner's complaint: "hand tracking triggers when it shouldn't; I need a
#   foolproof way to make it trigger every time I want it but with FEWER false
#   triggers." The fix is a HYBRID engage model with two modes (skills/
#   kinect_air_mouse.py :: engage_decision):
#     • PASSIVE (default): a STRICT smart-pose gate — the cursor is taken only on
#       an OPEN PALM, raised above the shoulder, FACING the sensor, held STILL for
#       a brief DWELL (~0.30 s). A natural fast reach passes through the zone
#       quicker than the dwell and never engages, so gesturing/stretching/reaching
#       no longer grabs the cursor.
#     • ARMED (opt-in via voice: "mouse control on" / "take the cursor"): a RELAXED
#       gate — height-only, held for a short debounce; grip/facing/stillness are
#       NOT required, because the owner explicitly asked for control so it should
#       be responsive. "mouse control off" disarms back to PASSIVE (it does NOT
#       disable the feature — KINECT_AIR_MOUSE_ENABLED above is the real master off).
#   DISENGAGE stays snappy in BOTH modes (drop below the down-margin, a sustained
#   closed fist while engaged, tracking-loss grace, real-input yield, or voice
#   disarm) — never harder to release than before.
# AIR_MOUSE_REQUIRE_OPEN_PALM — PASSIVE mode requires the debounced-stable grip to
#   be OPEN ("open palm") to engage. This is the headline false-trigger fix: a
#   closed/pointing hand reaching or gesturing no longer takes the cursor. True.
AIR_MOUSE_REQUIRE_OPEN_PALM = True
# AIR_MOUSE_ENGAGE_DWELL_SEC — the PASSIVE "brief hold": the full smart pose
#   (raised + open + facing + still) must be SUSTAINED this long before the cursor
#   is taken. A fast natural reach crosses the zone quicker than this and never
#   engages; a deliberate hold-to-grab does. ~0.30 s. The HUD priming ring fills
#   0→1 over this window (see the overlay `prime` key).
AIR_MOUSE_ENGAGE_DWELL_SEC = 0.30
# AIR_MOUSE_ENGAGE_STILL_M — the PASSIVE stillness bar: total hand travel (summed
#   3D displacement) across the dwell window must stay UNDER this to keep priming.
#   A hand that is still mid-reach (settling to point) primes; a hand sweeping
#   past does not. ~6 cm.
AIR_MOUSE_ENGAGE_STILL_M = 0.06
# AIR_MOUSE_FACING_MAX_DEG — the PASSIVE facing bar: the body must face the sensor
#   within this many degrees of square (|facing_yaw_deg| <= this) to engage. If the
#   bridge doesn't provide facing (older build / not measurable) this signal is
#   skipped GRACEFULLY — missing facing is treated as "facing OK" so it never
#   becomes an un-passable gate. ~40°.
AIR_MOUSE_FACING_MAX_DEG = 40.0
# AIR_MOUSE_ARM_RELAXES_GATE — when ARMED, use the RELAXED height-only gate (a
#   short debounce, no grip/facing/stillness/dwell). The owner said "be responsive
#   when I explicitly ask for control." True. Set False to keep the full smart
#   pose even when armed (armed then only skips the dwell-hold vs passive).
AIR_MOUSE_ARM_RELAXES_GATE = True
# AIR_MOUSE_ARM_ENGAGE_DEBOUNCE_SEC — the short hold the ARMED (relaxed) gate uses
#   before engaging, so a 1-frame height spike still can't grab it while a quick
#   deliberate raise engages almost instantly. ~0.15 s.
AIR_MOUSE_ARM_ENGAGE_DEBOUNCE_SEC = 0.15
# AIR_MOUSE_FIST_RELEASES — a SUSTAINED closed fist while engaged force-disengages
#   (an extra, optional snappy release to "let go" without lowering the hand).
#   DEFAULT OFF (2026-07-07 owner report): it FOUGHT the click/drag gesture — a
#   normal close to click/drag, held ~0.6 s, tripped the release and STOPPED
#   tracking ("when I close my hand it stops tracking"). With it off, closing the
#   hand clicks/drags and the cursor keeps tracking; you let go by LOWERING your
#   hand. Re-enable via the Settings panel if you want fist-to-release back.
AIR_MOUSE_FIST_RELEASES = False
# AIR_MOUSE_FIST_RELEASE_SEC — how long the fist must stay closed (while engaged)
#   before it counts as a release, so a normal click/drag (close→open, or a short
#   held drag) never trips it — only a deliberate sustained fist. ~0.60 s.
AIR_MOUSE_FIST_RELEASE_SEC = 0.60
# AIR_MOUSE_PER_APP_DISABLE — when True, the air-mouse STANDS DOWN (and force-
#   disengages if already engaged) whenever the FOREGROUND window's title/class
#   matches any AIR_MOUSE_DISABLED_APP_HINTS substring — e.g. a fullscreen game or
#   video where a stray cursor grab would be disruptive. Defensive: any win32
#   failure is treated as "not disabled" so it never accidentally kills the mouse.
AIR_MOUSE_PER_APP_DISABLE = True
# AIR_MOUSE_DISABLED_APP_HINTS — lower-case substrings matched against the
#   foreground window TITLE and CLASS name. Sensible defaults: common fullscreen
#   games / video players where the air-mouse should stay out of the way. Editable
#   in data/user_settings.json.
AIR_MOUSE_DISABLED_APP_HINTS = [
    "full screen", "fullscreen",           # generic fullscreen markers
    "netflix", "youtube - ", "prime video",  # streaming players
    "vlc media player", "mpc-hc", "kodi",  # desktop video players
    "steam big picture", "moonlight",      # game launchers / streaming
    "unrealwindow", "unitywndclass",       # common game engine window classes
]
# AIR_CONTROL_ENABLED — movie-style AIR CONTROL (skills/air_control.py, engine
#   core/air_control.py): reach a hand OUT toward the sensor + above the waist
#   to take the cursor across the WHOLE virtual desktop; a closed FIST grabs and
#   drags, a quick close→open is a click, a LASSO (pointing) hand scrolls, and
#   dropping/retracting the hand releases everything. Distinct from the
#   raise-above-shoulder KINECT_AIR_MOUSE above (different engagement model +
#   grab/scroll semantics); don't run both at once.
#   SAFETY — DEFAULT False: a Kinect hand-state glitch must NEVER drive the real
#   mouse uninvited, so nothing moves at boot. The SKILL still LOADS when this is
#   False (so the voice actions exist); the knob only controls whether the
#   control LOOP auto-starts at load. "Air control on" starts the loop
#   explicitly at runtime regardless of the knob (an explicit voice command IS
#   the owner's consent); "air control off" stops it and releases any held
#   button. Overridable via data/user_settings.json like every other flag.
AIR_CONTROL_ENABLED = False
# KINECT_HAND_MIRROR — the Kinect color/skeleton stream is MIRRORED (selfie view),
#   so the owner's REAL left hand appears on the RIGHT of the image. When True
#   (the default) the air-mouse SWAPS the bridge's left↔right hands so the owner's
#   REAL left hand = LEFT-click + left-side circle and their REAL right hand =
#   RIGHT-click. Flip False only if a future build un-mirrors the stream. See
#   skills/kinect_air_mouse.py (_hand_mirror_enabled / _mirror_sample).
KINECT_HAND_MIRROR = True
# KINECT_TWO_HAND_ENABLED — when True (the default), raising BOTH hands above the
#   shoulder (the same raise-to-engage lift gate the air-mouse uses) enters TWO-HAND
#   pinch-to-resize: GRAB the foreground window with both hands (held ~0.2 s), SPREAD
#   to grow / PINCH to shrink it about its centre (proportional to the 3D hand-
#   distance, EMA-smoothed), and move both hands together to translate it. Release a
#   hand to finish. While two-hand mode is active the single-hand air-mouse cursor
#   STANDS DOWN (so the two don't fight the cursor) and the HUD draws TWO reticle
#   circles (blue, purple while resizing). Targets only a normal foreground window;
#   the shell/desktop/taskbar are skipped. Never runs in staging/test. A ~30 Hz
#   background poller self-gates on this flag each tick (cheap to leave running when
#   off). See skills/kinect_two_hand.py (+ skills/kinect_air_mouse.set_two_hand_active
#   / two_hand_active for the hand-off, hud/jarvis_air_cursor.py for the reticles).
KINECT_TWO_HAND_ENABLED = True
# KINECT_GREET_ON_ENTRY — when True (and presence is enabled), JARVIS speaks a
#   brief varied greeting when you enter a room that had been empty for a while.
#   Hard rate-limited (≤ once/min) and skipped mid-conversation. Off by default.
KINECT_GREET_ON_ENTRY = False
# KINECT_POSTURE_NUDGE — when True (and presence is enabled), JARVIS estimates
#   slouch from the Kinect spine joints and tracks seated time, emitting ONE
#   gentle posture/stand nudge after a sustained hunch (~10 min) or long seated
#   stretch (~45 min), then cooling down (~20 min). Off by default; never nags.
KINECT_POSTURE_NUDGE = False
# KINECT_GUARD_ENABLED — when True, the owner may ARM guard mode (the multi-angle
#   security array in skills/guard_mode.py): an armed background daemon watches
#   every camera (both webcams via frame differencing + the Kinect's skeleton
#   presence) and, on detected motion/intrusion, snapshots the frame to a
#   gitignored data/guard_snapshots/ folder and fires ONE rate-limited proactive
#   alert (spoken + phone-push if configured). This flag only decides whether
#   arming is ALLOWED — arming is always an explicit voice action ('guard the
#   room' / 'stand down'), never automatic. Off by default.
KINECT_GUARD_ENABLED = False


# ─── Camera-based "is the TV on?" detector (ambient-suppression, opt-in) ─
# TV_DETECT_ENABLED — when True, a lightweight background detector periodically
#   samples the cached face-tracking camera frame and decides whether a TV/
#   monitor screen is visibly ON — a bright, FLICKERING (high frame-to-frame
#   temporal variance) rectangle — in a calibrated region (whole frame if not
#   calibrated). When it sees a live screen it contributes ONE MORE veto signal
#   to ambient-learning suppression: _ambient_media_is_playing() OR's it in, so a
#   TV the AUDIO gates miss (a muted TV, an unrecognised stream, a show the
#   content judge can't place) still stops JARVIS ingesting on-screen chatter as
#   the owner's facts. It is PURELY a SUPPRESSION signal — it can only veto
#   learning, never trigger anything — and reads a frame the face-tracker already
#   captured (no extra camera open). Calibrate the rectangle once with "calibrate
#   the tv region" (stored normalised in a SEPARATE gitignored data/tv_region.json
#   via JARVIS_TV_REGION_PATH — never user_settings.json). OFF by default; with it
#   off NO frame is read and ambient suppression is byte-identical to today.
#   Staging-safe (never drives anything regardless). Voice: 'turn on/off tv
#   detection', 'tv detection status', 'calibrate the tv region'. See
#   audio/tv_detect.py (pure stats + region store) + skills/tv_detect.py (wiring).
TV_DETECT_ENABLED = False


# ─── Face recognition (identity, opt-in; all default off) ──────────────
# JARVIS can recognise WHO is at the desk — pairing the Kinect's body COUNT
# with an actual identity from the two monitor webcams (the cameras closest to
# the user's face). It uses OpenCV's built-in face modules (YuNet detector +
# SFace recognizer, ONNX) — no dlib, no extra pip dependency. The ~38 MB SFace
# model and the ~232 KB YuNet model download once to a gitignored data/models/
# folder; the engine never raises if the download fails, it just stays off.
#
# PRIVACY: this is FACE BIOMETRICS. It is OFF by default and fully opt-in. The
# face embeddings live ONLY in a gitignored data/face_enroll.json (biometric
# PII — never committed, never shipped) and never leave the machine. Nothing is
# captured or stored unless you explicitly enroll ("learn my face").
#
# FACE_ID_ENABLED — master switch. When False (default) the face_id skill
#   refuses every action with an honest line and no camera/model work happens;
#   situational_awareness keeps its existing webcam+Kinect behaviour unchanged.
#   Set True to allow enrollment + recognition from the monitor webcams.
FACE_ID_ENABLED = False
# FACE_ID_MATCH_THRESHOLD — SFace cosine-similarity floor for a positive match.
#   rec.match(..., FR_COSINE) returns HIGHER for more-similar faces; OpenCV's
#   own SFace reference uses 0.363 as the same-person cutoff (a feature whose
#   best cosine vs an enrolled person is >= this is named, else "unknown").
#   Raise it to be stricter (fewer false matches), lower it to be more lenient.
FACE_ID_MATCH_THRESHOLD = 0.363
# GREET_NEW_PEOPLE_ENABLED — proactive "who are all these new people?" greeting.
#   When True (and FACE_ID_ENABLED is on so the webcams can actually recognise
#   faces), the face-tracker poller watches the primary webcam for MULTIPLE
#   UNRECOGNISED faces (people NOT enrolled in face-ID) held for a few seconds
#   and, once per gathering, fires ONE short varied proactive line — for when
#   the owner has friends over. The owner's own enrolled face never counts as
#   "new". Hard rate-limited (≤ once per ~10 min) and skipped mid-conversation,
#   exactly like KINECT_GREET_ON_ENTRY. OFF by default: with it off NO extra
#   recognition runs and behaviour is byte-identical to today. Flip on by voice
#   ('notice when people arrive' / 'say hi to guests') or here.
GREET_NEW_PEOPLE_ENABLED = False


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

# Live camera preview in the HUD — a small downscaled mirror of what JARVIS
# actually sees (the primary face-tracking frame; with KINECT_AS_CAMERA this is
# the Kinect 1080p color stream). The main process writes ONE overwriting,
# downscaled (~240px) JPEG to data/.hud_camera_preview.jpg a few times a second
# and the (separate-process) unified HUD loads + displays it in a corner.
# Privacy: exactly one temp file (never a growing folder); the main process
# STOPS writing it — and removes it — whenever the camera is off or face-
# tracking is paused, so the HUD falls back to a "CAMERA OFF" placeholder and no
# stale frame lingers on disk. Set False to disable the preview entirely (no
# JPEG is ever written).
HUD_CAMERA_PREVIEW = True

# Kinect SKELETON OVERLAY in the HUD camera preview (PART A). When True AND the
# Kinect is enabled + streaming, the HUD camera tile shows the KINECT COLOR frame
# with the LIVE tracked SKELETON drawn over it (bones between adjacent joints +
# dots at joints), composited with two small webcam tiles ('Fullhan Webcam' =
# left, 'USB 2.0 Camera' = right, resolved by name). This REPLACES the plain
# primary-webcam mirror in the preview JPEG with the richer Kinect+skeleton view;
# the same single .hud_camera_preview.jpg pipeline (atomic write + stale-file
# guard) carries it, so the separate-process HUD needs no Kinect / pygrabber of
# its own. It doubles as the owner's diagnostic that the body stream is live.
#
# Default False (staging-safe): with it off NOTHING changes — the preview stays
# the plain primary-camera mirror and no Kinect color/body frame is read for the
# HUD. Needs KINECT_ENABLED so the bridge actually opens the sensor; with the
# Kinect off it silently degrades to the normal webcam preview. Flip True here
# (or via the Settings GUI / user_settings.json) to see the skeleton.
KINECT_SKELETON_OVERLAY_ENABLED = False

# Full-virtual-screen translucent target reticle that flashes for ~2s wherever
# JARVIS performs a UI-automation action. KEPT ON — it is click-feedback, not
# an info widget, so it isn't part of the HUD clutter and is invisible except
# during the brief flash. It is click-through and never needs repositioning.
RETICLE_OVERLAY_ENABLED = True

# System-tray applet (tray.py at project root, pystray + Pillow).
TRAY_ENABLED = True

# ─── Live web interface (tools/web_interface.py + skills/web_interface.py) ──
# A local-LAN web dashboard to SEE what JARVIS is doing (live session-log tail,
# version / awake state / model routing / VRAM) and to TALK TO HIM BY TEXT — a
# typed command is fed through the EXACT SAME file-based inject channel a spoken
# command uses (injected_commands.json), so it behaves identically to voice.
#
# Default OFF: the server never binds a socket unless the owner opts in, so a
# fresh install exposes no new attack surface. When True, skills/web_interface.py
# auto-starts a daemon-thread http.server at boot on WEB_INTERFACE_BIND:PORT.
#
# SECURITY — LAN-EXPOSURE RISK. This endpoint can INJECT COMMANDS JARVIS EXECUTES
# (open apps, control the smart home, read the screen). Anyone who can reach the
# bound socket can drive JARVIS. Therefore:
#   • WEB_INTERFACE_BIND defaults to 127.0.0.1 (localhost only — nothing off-box
#     can reach it, no token needed).
#   • To expose it on the LAN (bind 0.0.0.0 or a LAN IP) you MUST set a non-empty
#     WEB_INTERFACE_TOKEN. The server REFUSES TO START on a non-local bind with an
#     empty token (it logs a clear reason and stays down), and when a token is set
#     it is required on EVERY request (Authorization: Bearer <token>, an
#     X-Auth-Token header, or ?token=… on the URL). Treat the token like a
#     password; anyone with it can command JARVIS from any device on your network.
# All four knobs are overridable via data/user_settings.json (the Settings GUI).
WEB_INTERFACE_ENABLED = False       # master switch — server only starts when True
WEB_INTERFACE_PORT    = 8766        # TCP port (8443 is the AirTag tracker — do NOT reuse)
WEB_INTERFACE_BIND    = "127.0.0.1" # bind address; non-local REQUIRES a token
WEB_INTERFACE_TOKEN   = ""          # shared secret; MANDATORY for a non-local bind

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

# ── Bambu chamber-camera HUD (hud/bambu_camera_hud.py) ──────────────────
# Master switch for the live printer-camera surface. When True, JARVIS can
# show the H2D's built-in camera in a movable HUD panel (voice: "show the
# printer camera"), and the frame grabber (core/bambu_camera.py) is allowed
# to pull frames over the LAN. The camera is fetched via the printer's
# authenticated LOCAL stream — RTSPS on port 322 for the H2D/X-class
# (requires "LAN Only Liveview" enabled on the printer screen), with a
# port-6000 JPEG-stills fallback for P1/A1-class printers. No Bambu Cloud
# round-trip. Reuses the existing BAMBU_PRINTER_IP / BAMBU_ACCESS_CODE
# credentials. Set False to disable the feature entirely (grabber + widget).
# Unlike the retired overlays above this is NOT auto-launched at boot — it's
# summoned on demand and (optionally) auto-shown while a print is active via
# BAMBU_CAMERA_AUTO_WHILE_PRINTING below.
HUD_BAMBU_CAMERA = True
# When True (and HUD_BAMBU_CAMERA is on), the camera panel auto-shows while a
# print is RUNNING/PAUSE/PREPARE and retires shortly after — same watcher
# pattern as the retired bambu corner overlay. Default False so the camera is
# opt-in / on-demand and never pops up unbidden.
BAMBU_CAMERA_AUTO_WHILE_PRINTING = False


# ─── Auto-switch default audio on headset power (audio/audio_switch.py) ──
# A USB-dongle wireless headset (e.g. a CORSAIR VOID ELITE) keeps its dongle
# plugged in whether the headset is ON or off, so plug/unplug detection misses
# the power state. Windows flips the headset's audio ENDPOINT Active<->NotPresent
# instead; this watcher polls that and moves the SYSTEM default render device:
#   headset ON  -> default = the headset   (remembers the prior default)
#   headset OFF -> default = the prior default, else AUDIO_AUTOSWITCH_FALLBACK
# Opt-in. HEADSET/FALLBACK are case-insensitive substrings of the Windows
# device friendly name (see `python -m audio.audio_switch --list`).
AUDIO_AUTOSWITCH_ENABLED  = os.getenv("JARVIS_AUDIO_AUTOSWITCH", "").lower() in ("1", "true", "yes", "on")
AUDIO_AUTOSWITCH_HEADSET  = os.getenv("JARVIS_AUDIO_HEADSET", "")    # e.g. "CORSAIR VOID ELITE"
AUDIO_AUTOSWITCH_FALLBACK = os.getenv("JARVIS_AUDIO_FALLBACK", "")   # e.g. "Realtek USB2.0 Audio"
AUDIO_AUTOSWITCH_POLL_S   = float(os.getenv("JARVIS_AUDIO_POLL_S", "3.0"))


# ─── Mic / speaker device selection (bobert_companion _refresh_devices) ─
# These live HERE (not in bobert_companion.py) so the Settings GUI mic-device
# picker -> data/user_settings.json -> _apply_user_settings() override path
# reaches them; bobert_companion.py consumes them via `from core.config import *`.
# Moved out of the monolith 2026-06 to fix the blocker where a written
# MICROPHONE_INDEX never reached the runtime (it was redeclared in the monolith
# AFTER the wildcard import, so the override was silently shadowed).
#
# PREFERRED_*_DEVICES — ordered lists of device-name substrings. Bobert picks
# the first connected match and auto-switches as devices come/go. Seeded from
# the JARVIS_PREFERRED_INPUT_DEVICES / _OUTPUT_DEVICES env vars (comma-separated
# substrings); empty default => use whatever the OS reports as default.
PREFERRED_INPUT_DEVICES  = [s.strip() for s in os.getenv("JARVIS_PREFERRED_INPUT_DEVICES", "").split(",") if s.strip()]
PREFERRED_OUTPUT_DEVICES = [s.strip() for s in os.getenv("JARVIS_PREFERRED_OUTPUT_DEVICES", "").split(",") if s.strip()]

# Manual overrides — set to an integer index to FORCE a specific device and
# disable auto-switching. None = use the PREFERRED_*_DEVICES lookup above. A
# NEGATIVE MICROPHONE_INDEX is the "hard-off / no mic" contract (staging green
# candidate, or the GUI's "Off (no mic)" choice): _mic_input_disabled() reads it
# so no capture stream is ever opened (it must NOT fall through to the system
# default mic). The picker persists the int (or null) — see settings_window.py.
MICROPHONE_INDEX = None
SPEAKER_INDEX    = None


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


# ─── Apple Music app (UWP) autostart + keep-alive ──────────────────────
# The Microsoft-Store Apple Music app (process AppleMusic.exe) has NO COM
# automation surface and NO system tray of its own, so JARVIS hosts the
# controls in ITS tray and drives playback only the LEGITIMATE way: launch
# the app via its AUMID and send OS media keys. These two opt-in flags let
# the user keep the app permanently running so those tray controls always
# have something to talk to. The keeper (audio/apple_music_keeper.py) reads
# them; it NEVER launches anything in staging/test. Both default False so a
# fresh install never pops the app open uninvited.
#
# APPLE_MUSIC_AUTOSTART — launch the Apple Music app once when JARVIS starts.
# APPLE_MUSIC_KEEP_OPEN — keep-alive: a background loop re-launches the app if
#   it gets closed (only ever (re)launches when it is NOT already running, so
#   it never steals focus on a tick where the app is already up).
APPLE_MUSIC_AUTOSTART = False
APPLE_MUSIC_KEEP_OPEN = False


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
# RAG_ENABLED — master switch for the personal-RAG autostart. When False the
# skill still registers its actions (rag_status etc.) but never indexes or
# loads the embedding model, so it consumes no VRAM. Exposed in the Settings
# GUI so the VRAM-budget bar can account for the ~0.3 GB nomic-embed-text load.
# RAG_EMBED_MODEL — Ollama embedding model. 'nomic-embed-text' runs on
# the 3090 at 200+ docs/sec.
RAG_ENABLED         = True
RAG_INDEX_PATHS = [
    os.path.join(os.path.expanduser("~"), "Documents"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.join(os.path.expanduser("~"), "OneDrive"),
]
RAG_EMBED_MODEL     = "nomic-embed-text"
RAG_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/embeddings"
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
# STANDBY_WHISPER_PREFER_GPU — load this loop's whisper-tiny on the GPU first.
# CPU by default to preserve VRAM for the local LLM: the resident 30B fills
# almost all of the 24GB, so a CUDA-resident whisper here competes for the last
# few hundred MB and has contributed to an OOM crash. Left False, the loop loads
# on CPU (int8 — whisper-tiny is cheap there). Set True only when there's VRAM
# headroom to opt back into the faster CUDA-first path (float16, frees a CPU
# core), which falls back to CPU/int8 if the GPU load fails.
STANDBY_WHISPER_PREFER_GPU        = False


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
    # Back-compat aliases: a config constant that was RENAMED still has saved
    # user_settings.json files (and Settings-GUI writes) carrying the OLD key.
    # Map the legacy name onto the current one so the user's override still
    # reaches the live constant instead of being silently dropped by the
    # `key not in g` guard below. 2026-06: AMBIENT_LISTENING_ENABLED was renamed
    # to AMBIENT_LISTEN_ENABLED in v1.20.0; the owner's file still had the old
    # key, so the mic-ambient daemon never autostarted ("not even learning").
    _LEGACY_KEY_ALIASES = {
        "AMBIENT_LISTENING_ENABLED": "AMBIENT_LISTEN_ENABLED",
    }
    g = globals()
    for key, val in data.items():
        key = _LEGACY_KEY_ALIASES.get(key, key)
        if key.startswith("_") or key not in g:
            continue          # only override an existing public config constant
        cur = g[key]
        # int|None knobs (current value None or a non-bool int) — e.g.
        # MICROPHONE_INDEX / SPEAKER_INDEX — accept an explicit null/blank as
        # "clear to None" (auto / system-default lookup). A None DEFAULT carries
        # no type for the isinstance ladder below to match, so without this the
        # saved value would be silently dropped: that was the MICROPHONE_INDEX
        # picker blocker (an int index written to user_settings.json never
        # reached the runtime). Numeric values coerce to int (a negative index
        # is the GUI's "Off (no mic)" hard-off contract); a non-numeric value
        # (e.g. "abc") raises and is skipped, leaving the default intact.
        _is_int_or_none = cur is None or (isinstance(cur, int)
                                          and not isinstance(cur, bool))
        try:
            if _is_int_or_none:
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    # Only the genuinely-nullable knobs (None default) clear to
                    # None; a real int knob keeps its value rather than becoming
                    # None on a malformed null write.
                    if cur is None:
                        g[key] = None
                    # else: leave the int knob untouched (no valid override).
                else:
                    g[key] = int(val)
            elif isinstance(cur, bool):
                # A JSON *string* boolean ("true"/"false") — which the Settings
                # GUI's coerce_value accepts and can write — would be mis-read by
                # a bare bool(val): bool("false") is True. Parse string truthiness
                # the same way settings_window.coerce_value does so the runtime
                # constant and the GUI never disagree. 2026-07-08.
                if isinstance(val, str):
                    g[key] = val.strip().lower() in ("1", "true", "yes", "on", "y")
                else:
                    g[key] = bool(val)
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
