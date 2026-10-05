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
SCREEN_VISION_MODEL   = "claude-sonnet-5-5"


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
# Which model embeds long-term memory for search (2026-10-02, opt-in).
#   "BAAI/bge-small-en-v1.5" (default) -> today's index, untouched.
#   "voyage-4-nano" -> voyageai/voyage-4-nano on the CPU, 512-d. On the
#     owner's 236 real facts it ranked the right one first 49/50 vs 39/50 for
#     bge-small, at +33 ms per recall; about 0.7 GB downloaded on first use.
# Switching rebuilds the index from the stored fact texts on a background
# thread: the old index keeps answering until the new one is complete, and
# is then kept as a .bak, never deleted. If the model cannot load, JARVIS
# logs one line and stays on bge-small. Applies on the next start.
MEMORY_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# The shipped value, captured before _apply_user_settings() can overwrite the
# public constant (same idiom as _SHIPPED_LOCAL_LLM_MODEL): lets a test pin
# "the default is today's model" on a box whose user_settings.json opted in.
_SHIPPED_MEMORY_EMBED_MODEL = MEMORY_EMBED_MODEL

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
# "confirm" as the next utterance) before executing. user_settings.json can
# ADD keywords but never remove these (see _SAFETY_LIST_BASELINE, 2026-10-01).
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
# claude-sonnet-5-5 (owner decision 2026-10-01): the current Sonnet, $2/$10 per
# MTok — the same price as Sonnet 5 and stronger. It thinks by default (effort
# `high` → ~13 s to the first answer token), so every call is shaped by
# core.llm_client.request_options (voice = effort low, ~1.2 s; max_tokens
# floor 2048). Chat stays LOCAL-first per MODEL_ROUTING; this is the
# cloud fallback / cloud route. Unattended deep jobs (deep code audit, overnight
# ideas) use claude-opus-5-5 ($4/$20) — never a voice path, it always thinks.
# The ceiling, claude-fable-5-1 ($10/$50), is selectable in the Settings GUI.
CLAUDE_MODEL = "claude-sonnet-5-5"
OLLAMA_MODEL = "llama3"

# The small, fast Claude model (2026-10-02): the ONE place its id is written.
# The notification sorter's and email triage's Claude leg, the briefing
# orchestrator's per-source workers (ORCHESTRATOR_WORKER_MODEL blank) and the
# self-diagnostic's API probe (when CLAUDE_MODEL is unset) all read it, so a
# swap is one line in user_settings.json: "CLAUDE_FAST_MODEL": "<model id>".
# Haiku 4.5 ($1/$5 per MTok). Anthropic's deprecations page (checked
# 2026-10-02) lists claude-haiku-4-5-20251001 Active, deprecated N/A,
# tentative retirement "Not sooner than October 15, 2026", with at least 60
# days' notice before any retirement.
CLAUDE_FAST_MODEL = "claude-haiku-4-5"

# Retired-model successors (2026-10-02). When Anthropic answers not_found for
# a Claude model (retired, or not available to this key), core.llm_client logs
# ONE line per model per session and, if this maps the model to a replacement,
# retries on it and keeps using it for the session; with no entry the feature
# falls back to its local path (the local brain / local vision / raw data) or
# says honestly that the model is gone. Keys and values are model ids; a
# snapshot id (…-20251001) also matches its alias entry. Empty by default —
# Anthropic names a replacement only when it deprecates a model. Example:
#   "CLAUDE_MODEL_SUCCESSORS": {"claude-haiku-4-5": "claude-sonnet-5-5"}
CLAUDE_MODEL_SUCCESSORS = {}

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
# `_get_local_llm_model()` resolves in this order: JARVIS_LOCAL_LLM_MODEL, then
# the owner's persisted pick (LOCAL_LLM_MODEL when it differs from
# _SHIPPED_LOCAL_LLM_MODEL below AND is installed), then the first installed
# entry of bobert_companion._LOCAL_LLM_PREFERENCE, then the first installed tag.
# Do NOT re-list the chain's tags here — read that tuple; its one sanctioned
# mirror is core.model_catalog._LOCAL_FAILOVER_TAGS. (An earlier copy of this
# comment named qwen3:30b-a3b as a chain member; it never was — the note under
# LOCAL_LLM_MODEL below calls it a text-only opt-in "max brain". That stale list
# is exactly what deferring to the tuple prevents.)
# The old dense qwen2.5:32b default
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
# The SHIPPED default, captured BEFORE _apply_user_settings() (bottom of this
# file) can overwrite the public constant from data/user_settings.json. The
# resolver (bobert_companion._get_local_llm_model) compares LOCAL_LLM_MODEL
# against this to tell "the owner picked a model" from "still the default" —
# capturing it HERE, once, avoids hard-coding the tag a second time anywhere
# else (the stale-duplicate bug class). The underscore prefix keeps it out of
# `from core.config import *` AND makes it un-overridable: the settings
# apply-loop skips `_`-prefixed keys. 2026-07-21 audit.
_SHIPPED_LOCAL_LLM_MODEL = LOCAL_LLM_MODEL
# 127.0.0.1, NEVER "localhost" (2026-07-12): Windows resolves localhost to
# ::1 first, and when Ollama listens only on IPv4 the ::1 attempt eats a
# measured, rock-steady ~2.05s before falling back — pushing every request
# just past _ollama_alive()'s 2s probe. Result: Ollama up and healthy while
# EVERY availability check said dead ("local vision unavailable"), and the
# self-heal kept restarting a server that was never down. Same pin applied
# to every other 11434 reference in the tree.
LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434"

# ─── Local prompt-prefix stability (2026-09-29) ────────────────────────
# The local model re-evaluates the WHOLE prompt (~3 s for ~12k tokens,
# measured) whenever the system prompt changes between turns; an unchanged
# prefix costs only the new turn (~1.25 s). LOCAL chat route only — the cloud
# route is unaffected.
# PROMPT_FREEZE_QUIET_S — while you and JARVIS are talking (a turn or reply
#   within this many seconds), the post-turn system-prompt rebuild (newly
#   learned facts / topics) is held back and applied ONCE after this much
#   quiet. Float; 0.0 applies every rebuild immediately (the old behaviour).
# LOCAL_PREFIX_REPRIME — after such a held-back rebuild changed the prompt,
#   send the model the new prefix while you are quiet (one tiny request,
#   only if the model is already loaded, never in game mode or mid-turn), so
#   your next turn starts warm.
# LOCAL_REPRIME_AT_BOOT_S — this many seconds after JARVIS starts, send
#   the model the conversation's prompt once (the same small re-prime, same
#   safeguards: only if the model is already loaded, never in game mode or
#   mid-turn), so your FIRST turn after a restart is as quick as the rest
#   (it re-read the whole ~13k-token prompt, ~3.3 s, measured 2026-09-29).
#   Float; 0.0 turns it off.
# Changes apply on the next start.
PROMPT_FREEZE_QUIET_S = 30.0
LOCAL_PREFIX_REPRIME = True
LOCAL_REPRIME_AT_BOOT_S = 20.0

# ─── Local background traffic (2026-09-29) ─────────────────────────────
# The local model keeps only its most recent request warm, so ANY background
# request (memory extraction after a turn, the ambient extractor, a Teams
# screenshot read by the vision model) between two of your turns makes the
# next turn re-read the whole ~12k-token prompt (~2.2 s measured). LOCAL chat
# route only.
# LOCAL_BACKGROUND_MAX_DEFER_S — while you and JARVIS are talking (the
#   PROMPT_FREEZE_QUIET_S window), non-urgent background local work waits,
#   queued in order, until you go quiet — but never longer than this many
#   seconds (then it runs between turns). Your own requests never wait.
#   Float; 0.0 turns the waiting off (the old behaviour).
# LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S — after such background work ran,
#   quietly re-send your conversation's prompt to the already-loaded model
#   (the same small re-prime as above, same safeguards) if you spoke within
#   this many seconds, so your next turn starts warm. Float; 0.0 turns it off.
# BACKGROUND_TAG_STRICT (speed plan R5, 2026-10-02) — the background callers
#   found later (the notification classifier's local fallback, Chappie's
#   daemon, the credits monitor, the scheduled evening briefing, the morning
#   chain's briefing / handoff) wait the same way only when this is True. False (the default) keeps today's timing: they
#   run at once and the log says "[bg-local] shadow <job> would defer" when
#   they would have waited. The first-tagged jobs (memory extraction, the
#   ambient extractor, the Teams check, ...) always wait.
# Changes apply on the next start.
LOCAL_BACKGROUND_MAX_DEFER_S = 120.0
LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S = 600.0
BACKGROUND_TAG_STRICT = False

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


# ─── Local-first for the Claude-only features (2026-10-02) ─────────────
# Features that build their own Claude call outside the chat path. Screen
# vision needs no key here: MODEL_ROUTING["vision"] = "local" already keeps it
# (and everything that looks through ask_vision) on the local vision model.
# The briefing orchestrator's switch is ORCHESTRATOR_BACKEND (its section).
#
# NOTIFY_SORTER_BACKEND — who classifies a toast no rule matched
#   (skills/notification_triage.py): "local_first" | "claude".
#   "local_first" (default): a one-word classification on the local brain;
#     Claude (CLAUDE_FAST_MODEL) only when the local brain is unavailable or
#     its answer is not one of the four labels — and only where the cloud gate
#     allows (AI_BACKEND claude + a key + chat not routed local), as before.
#   "claude": Claude first, the local brain as the fallback (the order before
#     2026-10-02).
#   On a local-only setup both answer identically: the cloud gate is closed,
#   so only the local brain ever sees the toast.
NOTIFY_SORTER_BACKEND = "local_first"
# BROWSER_AGENT_BACKEND — skills/browser_agent.py: "claude" | "local".
#   "claude" (default): browser tasks are driven by Claude whenever a key is
#     set (unchanged). "local": never send page content to Claude. There is no
#     local browser-driving model, so browser tasks then decline with an
#     honest line instead of running.
BROWSER_AGENT_BACKEND = "claude"
# CREDITS_CHECK_BACKEND — how check_credits (skills/credits_monitor.py) reads
#   the billing-page screenshot: "auto" | "local".
#   "auto" (default): through screen vision, routed by MODEL_ROUTING["vision"]
#     (unchanged). "local": always the local vision model, so the billing page
#     never goes to Claude; needs local vision to be usable
#     (LOCAL_VISION_FALLBACK on, or MODEL_ROUTING["vision"] "local").
CREDITS_CHECK_BACKEND = "auto"


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
# AMBIENT_STT_YIELD — the ambient mic and system-audio daemons hold a batch
#   back (instead of transcribing it) while you are speaking to JARVIS, so
#   they never make your own transcription wait behind them; held batches are
#   transcribed and kept, in order, once your sentence is done. False by
#   default. Set via user_settings.json; applies on the next daemon start.
AMBIENT_STT_YIELD = False

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

# ELECTRICITY_RATE_PER_KWH — what the owner pays per kilowatt-hour, used by
#   the running_costs action (core/running_costs.py) to turn the measured GPU
#   draw + estimated CPU draw x hours running into an electricity ESTIMATE.
#   A float so a user_settings.json override like 0.3 keeps its decimals.
ELECTRICITY_RATE_PER_KWH = 0.14


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
# Planner + merger on Claude Sonnet 5.5 at effort low (core.llm_client); the
# per-source workers are tiny summarisers (Claude Haiku 4.5). Not Opus 5.5:
# the brief is spoken, the planner has a 20 s timeout, and Opus 5.5 always
# thinks (~13-22 s to the first answer token). 2026-10-01.
# ORCHESTRATOR_WORKER_MODEL blank = CLAUDE_FAST_MODEL (2026-10-02), so the
# Haiku id lives in one place; set it to pin the workers to another model.
ORCHESTRATOR_PLANNER_MODEL      = "claude-sonnet-5-5"
ORCHESTRATOR_WORKER_MODEL       = ""
ORCHESTRATOR_MERGER_MODEL       = "claude-sonnet-5-5"
# ORCHESTRATOR_BACKEND (2026-10-02) — "claude" | "local".
#   "claude" (default): planner, workers and merger call Claude first and fall
#     back to the local model only when a Claude call fails (unchanged).
#   "local": they never call Claude — every stage takes its existing local
#     path (local Ollama planner / workers / merger, then the raw tool data),
#     so briefing data (inbox senders, news, system state) stays on this PC.
#     Slower: the stages queue on the one local GPU. Only matters where the
#     orchestrator runs at all — on a local-only backend (AI_BACKEND=ollama or
#     chat routed local) there is no fan-out and the normal local turn
#     answers the briefing either way.
ORCHESTRATOR_BACKEND            = "claude"
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

# ─── Per-sentence speech (Kokoro) ──────────────────────────────────────
# When True and TTS_BACKEND is 'kokoro' (voice clone off), a long reply with
# more than one sentence starts playing its first sentence as soon as that
# sentence is rendered, while the rest renders in the background -- instead of
# rendering the whole reply before the first word. Same voice, speed and
# prosody preset for every sentence; an interrupt stops the remaining
# sentences; muted stays silent. Short replies (under ~120 characters), single
# sentences, 'wry' deliveries and every other backend are voiced whole, exactly
# as before. Splitting is conservative (never inside "3.5", "e.g.", "Mr.",
# "2:30 p.m." or an ellipsis) -- see core/sentence_tts.py. Changes apply on
# the next start.
SENTENCE_TTS_ENABLED = True

# ─── Kokoro engine speed (speed plan R4, core/kokoro_tts.py) ───────────
# KOKORO_PERSISTENT_PHONEMIZER: phonemize every line on ONE espeak backend
# built with the engine, instead of kokoro_onnx's stock call, which builds a
# fresh backend per line (~115 ms) and leaves four copies of the espeak-ng dll
# in %TEMP% each time (measured 2026-10-02). With it on, the whole process
# leaves one copy. Phonemizing and rendering are each one-at-a-time; a
# render that cannot start within the Kokoro synth timeout falls back to the
# edge voice. Any error switches back to the stock call for the session.
# KOKORO_RENDER_CACHE: keep finished renders in memory, keyed by a hash of
# model, voice, language, speed and text (the text itself is never stored).
#   'off'    → no cache (today's behaviour)
#   'shadow' → fill the cache and log would-hit / would-miss, serve nothing
#   'on'     → a repeated line plays from the cache instead of re-rendering
# KOKORO_RENDER_CACHE_MB caps the memory (least recently used goes first);
# KOKORO_RENDER_CACHE_PERSIST also keeps renders as .npy files under
# data/tts_cache/ (same cap) so they survive a restart. All OFF until proven.
# Changes apply on the next start.
KOKORO_PERSISTENT_PHONEMIZER = False
KOKORO_RENDER_CACHE          = "off"   # 'off' | 'shadow' | 'on'
KOKORO_RENDER_CACHE_MB       = 64
KOKORO_RENDER_CACHE_PERSIST  = False


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
# Engine id: "chatterbox" (in-process, below) or "chatterbox_turbo_server"
# (its own process; see the clone voice server block below).
VOICE_CLONE_MODEL   = "chatterbox"
# Torch device for the clone engine. "" = historical default (cuda:0 if present
# else cpu). Set "cuda:1" to run chatterbox on a SECOND, idle GPU so it stops
# eating the primary card's VRAM (frees ~3GB on the 3090 for the LLM). Gated by
# a free-VRAM check in core/voice_clone (degrades to the edge-tts ladder if the
# chosen device lacks room), so a bad value never OOMs. 2026-07-09.
VOICE_CLONE_DEVICE  = ""

# ─── Clone voice SERVER (VOICE_CLONE_MODEL = "chatterbox_turbo_server") ──
# 2026-10-03. The clone model runs in its OWN process and venv (CUDA torch on
# the RTX 3090) and JARVIS talks to it over the loopback
# (core/clone_voice_client.py), so nothing heavy loads inside JARVIS. Used
# when VOICE_CLONE_ENABLED is on, VOICE_CLONE_MODEL is
# "chatterbox_turbo_server" and VOICE_CLONE_PROFILE names a consented profile:
#   * boot: JARVIS reuses a server already answering at VOICE_CLONE_SERVER_URL,
#     else starts VOICE_CLONE_SERVER_CMD detached and waits for it (bounded,
#     off the boot path). If it never comes up: one log line, Kokoro speaks.
#   * each sentence is one POST /tts bounded by VOICE_CLONE_TIMEOUT_S (plus
#     0.03 s per character past 80) -- the budget of a line the listener is
#     waiting for. A sentence rendered ahead while earlier ones still play
#     may take until it is needed, and once a reply speaks in the clone
#     voice, up to its budget past that moment (a pause, not a change of
#     voice; 2026-10-04). A failed or slow line is voiced by Kokoro at once;
#     3 latency-critical misses in a row rest the clone for 5 minutes
#     (doubling each time, up to 30), then it is tried again, and one more
#     miss right after a rest starts the next one. A long first sentence is
#     split at a clause so the first word comes sooner.
#   * the server is used only while its voice prompt is the active consented
#     profile's reference.wav (compared by hash).
#   * per-sentence speech, the processing filler and the R3 pre-render work
#     with it as they do with Kokoro (TTS_BACKEND 'kokoro' is the fallback).
#     Prosody presets: the gain and the wry pause are honoured; rate and pitch
#     are not (the model has no speed or pitch control).
# VOICE_CLONE_SERVER_CMD is a command line (or a JSON array of arguments) and
# is the owner's to set, because it names his own paths. {ref} becomes the
# active profile's reference.wav and {port} the URL's port, e.g.
#   <venv>\Scripts\python.exe -u <dir>\clone_tts_server.py --ref {ref} --port {port}
# Empty = JARVIS never starts a server, but uses one that is already running.
# The server keeps running when JARVIS exits or restarts (the next boot reuses
# it) and holds ~2.5 GB of VRAM on the 3090 until it is stopped (POST
# /shutdown). Changes apply on the next start.
VOICE_CLONE_SERVER_URL = "http://127.0.0.1:8767"
VOICE_CLONE_SERVER_CMD = ""
VOICE_CLONE_TIMEOUT_S  = 2.5


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

# When True (default), JARVIS ignores spoken commands while music is playing -
# media the PC itself is playing (the Windows media session: Spotify, Apple
# Music, Chrome...) or SUSTAINED room music heard through the mic - UNLESS they
# start with the wake word "JARVIS", so it doesn't reply to song lyrics it
# overhears. Set False if it keeps cutting YOU off while your own music plays;
# you can then talk to it normally over the music. (Covers both since
# 2026-10-01; the media-session check used to ignore this switch.)
AMBIENT_MUSIC_REFUSE_WAKE = True

# Manual "wake-word mode" (Alexa-style): when True, JARVIS ignores EVERY spoken
# command that doesn't start with the wake word "JARVIS" — one utterance, one
# command. Off by default; the user flips it on by voice (e.g. for an external
# TV the OS media session can't see). _apply_user_settings() overrides this from
# user_settings.json, and the voice action toggles it live at runtime.
REQUIRE_WAKE_MODE = False

# Guest mode (2026-10-02): visitors are in the room. JARVIS answers every turn
# as usual but writes NOTHING to long-term memory -- no learned facts,
# projects, topics, episodes, session summaries or voice-command log entries
# -- until it is turned off. Off by default; flipped by voice ("guest mode
# on", "we have guests", "the guests have left") or the web dashboard, and
# the flip is saved here through user_settings.json like REQUIRE_WAKE_MODE,
# so it stays on across restarts until turned off. The live flag is
# core/guest_mode.py, seeded from this at boot (bobert_companion
# _guest_mode_boot).
GUEST_MODE = False

# Follow-up window for wake-word mode, in seconds. After the user addresses
# JARVIS by its wake word, follow-ups inside this window need no wake word, and
# each admitted one extends it. 0 (default) = strict wake-word mode, unchanged.
# Ported 2026-09-28 from the Dell edge node, which ran it at 45. CAUTION: every
# admitted utterance extends the window, so in a noisy room (the reason to use
# wake-word mode at all) steady crosstalk such as a TV can hold it open; see
# core/followup_window.py. Set via user_settings.json.
FOLLOWUP_WINDOW_S = 0.0

# Owner speech vocabulary (core/stt_vocab.py). STT_HOTWORDS: comma-separated names
# Whisper should expect (faster-whisper "hotwords"). STT_REPLACEMENTS: phrases Whisper
# keeps mishearing -> what was said ({"a cello": "Accelo"}), whole words, every
# transcript. SITE_SHORTCUTS: {"name": "https://..."} so "open <name>" (also on a
# monitor) opens that page. All empty by default; set in user_settings.json.
# STT_HOTWORDS also applies LIVE: an edit to user_settings.json after start reaches
# the next transcription (core/stt_vocab.live_hotwords); the other two need a restart.
STT_HOTWORDS = ""
STT_REPLACEMENTS: dict = {}
SITE_SHORTCUTS: dict = {}

# Owner-only learning (core/learn_gate.py). When True, JARVIS learns facts,
# topics and projects only from turns that were clearly the owner's: typed,
# led by the wake word (or right after a standby wake), matched to the
# owner's enrolled voiceprint, or a follow-up within LEARN_FOLLOWUP_S of one
# of those. A voice that scores below LEARN_VOICE_REJECT_BELOW against the
# enrolled voiceprint never teaches -- a MATCHED one too, the owner's own
# voice included (2026-10-01), so keep it under the owner's usual match
# score (voice_id names a speaker from 0.72). Overheard speech teaches only
# with a matched voiceprint, and the background ambient extractor stops. Off by
# default: without an enrolled voiceprint or the wake word, almost nothing
# would teach. Set via user_settings.json; applies on the next start.
LEARN_ONLY_FROM_OWNER = False
LEARN_FOLLOWUP_S = 90.0
LEARN_VOICE_REJECT_BELOW = 0.60

# Media gate (core/media_gate.py, 2026-10-01). Live 22:28:50 an Instagram reel
# playing on this PC said "Jarvis, find me a restaurant ... build a website" and
# JARVIS ran it. While another app on the PC is producing sound (an active
# playback session's peak meter at or above MEDIA_VOICE_GATE_PEAK, 0..1; the OS
# media session's "playing" when the meter can't be read), a mic turn whose
# voice scores below MEDIA_VOICE_GATE_REJECT_BELOW against the enrolled owner
# voiceprint is dropped ("[media-gate] PC audio playing and not the owner's
# voice"). The reel scored 0.43. The owner's own commands OVER media scored
# 0.48-0.52 on the same buffer (21:43:17, 22:59:47, 13:57:08 - 2026-10-02
# review repair), so the floor is 0.45, a short media control ("pause", "next
# song", "turn it down") always runs, a long capture is scored again on its
# leading speech, and a dropped "Jarvis, ..." gets one short spoken cue.
# Nobody enrolled / voice-ID unavailable -> allowed and logged. Typed turns, a
# stop word and guest mode always pass. Set via user_settings.json (Settings,
# Hearing); applies on the next start.
MEDIA_VOICE_GATE_ENABLED = True
MEDIA_VOICE_GATE_PEAK = 0.01
MEDIA_VOICE_GATE_REJECT_BELOW = 0.45

# Known-device speech filter (core/device_speech_filter.py). When True, an
# utterance that matches a line a known device speaks (phrase lists in the
# gitignored data/device_phrases/*.json) is ignored before the wake gate, the
# LLM and any learning, and can never wake JARVIS. A stop word always gets
# through. No phrase files = no filtering. Set via user_settings.json; applies
# on the next start.
DEVICE_SPEECH_FILTER_ENABLED = True

# Self-echo filter (core/self_echo.py): JARVIS never answers his own voice.
# A mic transcript captured while one of his own lines was playing (a line
# spoken from another thread — tray, timer, background announcement — while
# the main loop was listening), or whose speech began within SELF_ECHO_TAIL_S
# seconds after such a line ended, is ignored; so is a transcript that matches
# a line he spoke in the last SELF_ECHO_WINDOW_S seconds. A stop word always
# gets through, and so does the wake word unless his own line said it. Typed
# commands are never checked. Set via user_settings.json; applies on the next
# start.
SELF_ECHO_FILTER_ENABLED = True
SELF_ECHO_WINDOW_S       = 20.0
SELF_ECHO_TAIL_S         = 0.8

# Presence hold for queued proactive speech (core/owner_presence.py). Live
# 2026-10-01 the wellness nudge, the credits nag and a GPU pulse were spoken
# into an empty room (he left at 18:27), and a wellness nudge talked over
# people mid-conversation (21:16). The speech-queue drain now holds every line
# except his own reminders (timer / schedule / promise) and the guard alert
# while he is away or the room is talking. "Here" = an owner MIC turn within
# OWNER_PRESENT_VOICE_WINDOW_S, a sustained face within
# OWNER_PRESENT_FACE_WINDOW_S, or PHYSICAL keyboard / mouse input (injected
# input never counts) within OWNER_PRESENT_INPUT_WINDOW_S. Non-wake speech
# captured within ROOM_TALK_HOLD_S holds the queue too. When he is back, a
# held status line older than PRESENCE_STALE_STATUS_S is folded into one short
# "While you were away" recap instead of being read out on its own.
# PRESENCE_HOLD_ENABLED False = the drain speaks everything at once, as
# before. Set via user_settings.json; applies on the next start.
PRESENCE_HOLD_ENABLED        = True
OWNER_PRESENT_VOICE_WINDOW_S = 600
OWNER_PRESENT_FACE_WINDOW_S  = 120
OWNER_PRESENT_INPUT_WINDOW_S = 300
# The input watcher's polling fallback (the low-level hook is off by default,
# AIR_MOUSE_LL_HOOK_ENABLED) counts injected input - an automation driving the
# PC - like his. Polled input is presence only when his voice or a face was
# seen within this many seconds (2026-10-02).
OWNER_PRESENT_POLLED_INPUT_BACKUP_S = 900
ROOM_TALK_HOLD_S             = 45
PRESENCE_STALE_STATUS_S      = 600

# Device dialogues (core/dialogue.py): a skill that owns a talking device can
# run a short scripted back-and-forth between JARVIS and it through the
# skill_utils "dialogue_session" / "speak_line" / "listen_for_stop" hooks.
#   DIALOGUE_ENABLED     False refuses every dialogue ("disabled").
#   DIALOGUE_MAX_S       hard cap on one dialogue's length, seconds.
#   DIALOGUE_STOP_LISTEN False = no stop-listen capture after device lines
#                        (the tray, the wake word and the device still stop).
#   DIALOGUE_BEAT_S      pause after each device line: comic timing, and the
#                        window in which the owner's "stop" is heard.
#   DIALOGUE_LOST_HOLD_S how long proactive speech and non-wake mic turns are
#                        held after a dialogue ends because the device went
#                        away (the owner is probably talking to the device).
#                        Read by the monolith's _dialogue_session at each
#                        dialogue's end (0 = no hold, capped at 120).
# Set via user_settings.json; applies on the next start.
DIALOGUE_ENABLED = True
DIALOGUE_MAX_S = 40
DIALOGUE_STOP_LISTEN = True
DIALOGUE_BEAT_S = 0.6
DIALOGUE_LOST_HOLD_S = 12.0

# Skill utterance routes (bobert_companion._UTTERANCE_ROUTES): a skill can
# claim an exact request ("talk to the <device> about pizza") BEFORE the LLM,
# so the action runs instead of the model answering with chat. False sends
# every request to the LLM again. Set via user_settings.json; applies on the
# next start.
SKILL_ROUTES_ENABLED = True

# Noise heard as speech (core/speech_filter.hallucination_verdict, R10). When
# True, a mic transcript that is ONLY a classic Whisper hallucination ("Bye.",
# "Thank you.", "You", "Thanks for watching") is ignored as noise — logged
# "[noise] ignored", never the text — when Whisper's own confidence is poor,
# the capture was barely above the VAD threshold, or the owner has not spoken
# for a while and it is not an answer to something JARVIS just asked. A real
# short reply ("thank you" right after JARVIS answered, "bye" ending a
# conversation) is kept. Typed commands are never checked. Thresholds:
# SPEECH_FILTER_OVERRIDES (NOISE_*). Set via user_settings.json; applies on
# the next start.
NOISE_FILTER_ENABLED = True


# ─── Whisper STT (faster-whisper preferred, GPU when present) ──────────
# `WHISPER_DEVICE = 'auto'` lets ctranslate2 + torch decide; 'cuda'
# forces GPU 0; 'cuda:N' pins STT to a specific GPU (e.g. 'cuda:1' to run
# Whisper on a second card and keep the primary free for the LLM/voice);
# 'cpu' forces the legacy path. large-v3-turbo on the 3090 runs ~15× real-time
# at near-identical accuracy to large-v3. 'listen' = the listen card
# (LISTEN_GPU below). bobert_companion numbers GPUs by PCI bus
# (CUDA_DEVICE_ORDER=PCI_BUS_ID, NVML's order). An index is NOT an identity: a
# card added on a LOWER bus number than the 1650 (bus 8 here — any CPU-attached
# slot) takes 'cuda:1' and the 1650 becomes 'cuda:2'. To keep a model on the
# 1650 whatever is added, use 'listen' with LISTEN_GPU = its UUID or '1650'.
WHISPER_DEVICE      = "auto"            # "auto" | "cuda" | "cuda:N" | "cpu" | "listen"
WHISPER_MODEL_CUDA  = "large-v3-turbo"  # ~3.1 GB VRAM, 8x faster than large-v3
WHISPER_MODEL_CPU   = "small"           # CPU-friendly default when no GPU

# Speed plan R11 - decode knobs for the owner's turns (faster-whisper path).
# no_speech_prob is always 0.0 here, so faster-whisper's silence exit never
# fires and a low-logprob decode walks all six temperatures: the 4-10 s tails.
# WHISPER_TEMPERATURES None = faster-whisper's own ladder (unchanged); e.g.
# [0.0, 0.2, 0.4] caps it. WHISPER_BEAM_SIZE 1-10 (5 = unchanged). A bad value
# falls back to the default. The bounded no-VAD retry keeps beam 1 / temp 0.
WHISPER_TEMPERATURES = None
WHISPER_BEAM_SIZE    = 5


# ─── Parakeet STT on the CPU (speed plan R6, core/stt_parakeet.py) ─────
# STT_ENGINE: which engine decodes YOUR spoken commands (the owner's captures
#   only; ambient listening and in-turn captures always use Whisper, which
#   stays loaded as the fallback).
#   'whisper'  → today's behaviour
#   'parakeet' → NVIDIA Parakeet TDT 0.6B v2 (int8 ONNX, CPU only): roughly
#                0.15-0.3 s per command instead of ~1.7 s. It ignores
#                STT_HOTWORDS; a transcript that comes back empty, or that
#                the wake-word gates would drop for want of "JARVIS"
#                (wake-word mode, music, the post-dialogue hold, standby or
#                sleep) while the owner started talking at once, is decoded
#                again by Whisper.
#                Any error switches back to Whisper for the session.
#   The JARVIS_STT_ENGINE environment variable overrides this setting.
# STT_SHADOW: '' (off) | 'parakeet' → Whisper keeps transcribing; Parakeet
#   re-decodes each command (and each standby wake check) afterwards, while
#   JARVIS is idle, and the speech gates' verdicts on both go to the
#   gitignored data/stt_ab.jsonl. The words are written there only — never
#   to the log — and only for a line JARVIS would act on (a line no gate
#   would pass keeps its numbers, never its words); nothing is recorded
#   while the mic is muted; audio is never saved. Use it to compare the two
#   before switching.
# PARAKEET_MODEL_DIR: the downloaded model folder. PARAKEET_THREADS: CPU
#   threads for one decode. PARAKEET_CONF_ANCHORS: [[parakeet token
#   log-probability mean, Whisper-scale avg_logprob], ...] — how Parakeet's
#   confidence is mapped onto the scale the speech filter's thresholds use
#   (empty = the built-in calibration). STT_REPLACEMENTS_PARAKEET: like
#   STT_REPLACEMENTS ({"misheard": "meant"}), applied after it, to Parakeet's
#   transcripts only. All OFF by default; changes apply on the next start.
STT_ENGINE                 = "whisper"   # 'whisper' | 'parakeet'
STT_SHADOW                 = ""          # '' | 'parakeet'
PARAKEET_MODEL_DIR         = r"C:\JARVIS-models\parakeet-tdt-0.6b-v2-onnx"
PARAKEET_THREADS           = 8
PARAKEET_CONF_ANCHORS: list = []
STT_REPLACEMENTS_PARAKEET: dict = {}

# ─── Where the listening models run (core/listen_devices.py, 2026-10-04) ──
# PARAKEET_DEVICE / SMART_TURN_DEVICE / VOICE_ID_DEVICE: 'cpu' | 'listen' |
#   'cuda:N' (Whisper's is WHISPER_DEVICE above). 'listen' = the card
#   LISTEN_GPU names. A card that is missing, lacks the model's GPU runtime
#   (onnxruntime-gpu for Parakeet / Smart Turn; it is not installed) or has
#   less than the model's need + LISTEN_GPU_RESERVE_MB free puts the model on
#   the CPU with one "[listen]" log line. The defaults are what measured best
#   here: Parakeet 106-194 ms a command and Smart Turn 25 ms on the CPU;
#   voice-ID 10-75 ms on the CPU (on the 3090 it had taken a CUDA context and
#   up to 441 MB beside the brain).
# LISTEN_GPU: which card 'listen' means — 'cuda:N', a GPU UUID ('GPU-...',
#   printed on the boot "[listen] devices:" line; a unique prefix will do) or
#   a piece of its name ('1650'); '' = no listen card. 'cuda:1' is an index,
#   not a card: a card added on a lower PCI bus takes it. Name the card (its
#   UUID) to pin it. When a second RTX 3090 arrives, set this to its UUID to
#   move every 'listen' model there.
# All apply on the next start.
LISTEN_GPU                 = "cuda:1"
LISTEN_GPU_RESERVE_MB      = 512
PARAKEET_DEVICE            = "cpu"
SMART_TURN_DEVICE          = "cpu"
VOICE_ID_DEVICE            = "cpu"

# ─── Music gate (core/music_gate.py, 2026-10-04) ───────────────────────
# With music playing in wake-word mode every capture runs to 30 s and was
# transcribed twice (Parakeet, then a Whisper "rescue"), and the ambient
# listener ran Whisper on every 2.5 s of lyrics: ~35 % of the 1650 and ~39
# CPU-s a minute. MUSIC_GATE_MODE:
#   'off'    today's behaviour, nothing measured;
#   'shadow' (default) today's behaviour, plus one "[music-gate]" counter
#            line per minute with music: what 'on' would have skipped and
#            whether any of it would have been an owner turn ("lost");
#   'on'     over music: the ambient listener transcribes nothing, and
#            Parakeet's rescue runs only with a "Jarvis"-like word or the
#            owner's voice behind it (which also shortens the deaf gap
#            between captures: ~2 s instead of 5-6 s after a lyric capture).
# Wake-word and owner-voice detection (Parakeet on every capture, the media
# gate's voice check) keep running in every mode. No mode shortens a capture:
# nothing is recorded between captures, so more, shorter captures over music
# would mean more deaf gaps (core/music_gate.py).
MUSIC_GATE_MODE            = "shadow"

# Per-install speech-filter tuning. The Whisper gate thresholds depend on the
# MICROPHONE, so an install whose mic differs from the desktop's overrides them
# here (via user_settings.json) instead of editing core/speech_filter.py and
# carrying a local patch. Allowed keys: WHISPER_MIN_WORDS, WHISPER_TRUST_RMS,
# WHISPER_MIN_AVG_LOGPROB, WHISPER_MAX_NO_SPEECH_PROB, and the noise gate's
# NOISE_RMS_MARGIN, NOISE_OWNER_IDLE_S, NOISE_REPLY_WINDOW_S (R10); anything
# else is ignored.
# Empty (default) = the built-in thresholds. Example for a laptop mic:
#   {"WHISPER_MIN_WORDS": 3, "WHISPER_MIN_AVG_LOGPROB": -1.15, "WHISPER_TRUST_RMS": 0.15}
SPEECH_FILTER_OVERRIDES = {}


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


# ─── Processing filler ("Just a moment, sir." while a voice turn thinks) ──
# Owner request 2026-09-06. When a SPOKEN command is still being processed
# PROCESSING_FILLER_DELAY seconds after the transcript was accepted and
# nothing has been said yet, JARVIS says one short pre-rendered butler line.
# After PROCESSING_FILLER_STILL_DELAY seconds of turn silence he says one
# "still working" line (once per turn). Typed / web / injected turns never get
# one; nor do muted, standby, focus / DND, night-owl or game mode, stop /
# cancel commands, a live in-turn mic capture or a running wake-word
# barge-in listener. Clips are rendered on the CPU Kokoro backend only
# (TTS_BACKEND='kokoro', voice clone off); on any other backend the filler is
# simply unavailable. The first voice turn after a start renders the clips.
# All logic: core/processing_filler.py. OFF by default until proven by ear.
# Delays are floats (an int default would make _apply_user_settings truncate
# a saved 2.5 to 2) and are clamped in processing_filler.sanitize_delays; a
# STILL_DELAY at or below the DELAY turns the "still working" line off.
# Changes apply on the next start.
PROCESSING_FILLER_ENABLED     = False
PROCESSING_FILLER_DELAY       = 2.5    # s after the transcript before stage 1
PROCESSING_FILLER_STILL_DELAY = 12.0   # s of turn silence before stage 2

# Filler handoff (speed plan R3, 2026-10-02). Every default below is today's
# behaviour; the owner flips them in user_settings.json (next start).
# PROCESSING_FILLER_LATE_START_S — stage 1 retries this long past its delay
# while a capture holds it off, and never STARTS later than that (nor later
# than the fixed 1 s cap from 2026-10-02, so only values under 1.0 change
# anything). 0.6 is the plan's pick; clamped to 0.1-60.
# PROCESSING_FILLER_SKIP_PLEASANTRIES — no filler for a turn that is only
# "thank you" / "hello" / "okay" / "good night" ... (owner's choice).
# FILLER_DUCK_HOLD — keep the music ducked from the filler clip through the
# answer to the end of the turn (one session scan, no swell in between).
# PROCESSING_FILLER_PRERENDER — while the filler clip plays, render the
# answer's first audio on the turn's own thread (Kokoro only, never under the
# speech lock); it plays only if it is exactly what the normal path would
# render, else the normal path runs. The turn line then shows pre=1.
PROCESSING_FILLER_LATE_START_S      = 3.0
PROCESSING_FILLER_SKIP_PLEASANTRIES = False
FILLER_DUCK_HOLD                    = False
PROCESSING_FILLER_PRERENDER         = False


# ─── Answer first (skip the model's short lead-in before a spoken answer) ──
# 2026-09-29. A reply like "One moment, sir. [ACTION: get_time]" used to speak
# "One moment, sir." in full and only THEN the action's real answer, so the
# answer arrived 1.5-2.5 s later (and a lead-in such as "I'll have to check on
# that" could contradict it). With this on, a SHORT lead-in that is pure
# acknowledgement (15 words or less, no digits, no question, nothing but
# "one moment" / "on it" / "let me check" style words) is not spoken when every
# action in the reply speaks a real answer: a verbatim-result action produced a
# short one, or an informative action runs a follow-up round while the
# processing filler covers the wait. A lead-in that carries content, or that
# confirms a side-effect action, is always spoken. The lead-in stays in the
# conversation history. Pushback / confirmation / hallucination replacements
# are never touched. Changes apply on the next start.
ANSWER_FIRST_ENABLED = True

# ─── Turn-timing telemetry (speed plan R1, 2026-10-01) ────────────────────
# Print-only fields on each turn's [turn-timing] line; neither changes what
# JARVIS hears, says or when (field meanings: core/turn_timing.py).
# TURN_TAIL_PROBE — measure where the owner's speech really ended: a Silero
# speech detector (core/endpointing.py, its own ~2 MB CPU session) runs over
# the last few seconds of each captured clip on a background thread and the
# line gets tail_ms. Latches off for the session on the first failure (one
# log line). TURN_PLAY_OPEN_PROBE — time the answer's first playback from
# entering the playback body to the started stream (play_open_ms) and note
# that stream's reported output latency (out_lat_ms). Set via
# user_settings.json; apply on the next start.
TURN_TAIL_PROBE = True
TURN_PLAY_OPEN_PROBE = True

# ─── Playback keeper (2026-10-05) ─────────────────────────────────────────
# Opening the speaker for a clip took ~350 ms on most of the owner's turns
# (play_open_ms p50 ~377) and under 50 ms whenever other audio was playing on
# the same speaker. Silent check on his speaker, 2026-10-05: 10/10 sd.play()
# opens 320-346 ms with nothing else open, 10/10 at 3.6-9.7 ms with a
# zero-filled stream held open on it. PLAYBACK_KEEPER 'on' plays SILENCE on
# the speaker JARVIS uses for the length of one reply: a zero-filled stream
# opened when a turn is answered or a line is about to be spoken, closed
# LINGER_S (2 s) after the last line; so the first line and every sentence
# after it open fast. Never permanent (it yields to the hotplug
# re-enumeration, and one holder counts for 60 s at most); see
# core/playback_keeper.py. 'on' also polls the playback reaper every 10 ms
# instead of 50 (the gap after each sentence and the barge-in cut). 'off' =
# exactly the old path. Applies on the next start.
PLAYBACK_KEEPER = "on"
# PLAYBACK_PRIMED_STREAM — each line plays on JARVIS's own stream instead of
# sd.play()'s: the line itself fills the speaker's ~0.2 s start-up buffer
# (sd.play() fills it with silence, so every line starts ~0.2 s late), and
# the stream ends by playing out what is queued (sd.play() discards it: the
# last ~0.15 s of each line, or the pause after a sentence, is never heard).
# Same reaper, barge-in and device rules (bobert_companion._open_primed_
# stream). It changes what you hear, so it is OFF until judged by ear.
# out_lat_ms on the turn line stays the stream's reported latency; with this
# on the first sample is not behind it. Applies on the next start.
PLAYBACK_PRIMED_STREAM = False

# ─── Smart Turn end of turn (speed plan R7, 2026-10-02) ───────────────────
# record_speech ends every turn after the same 21 silent chunks (1,344 ms),
# finished sentence or mid-thought pause alike. Smart Turn v3.2 (an ~9 MB
# audio model, pipecat-ai/smart-turn-v3) hears the capture's last 8 s and says
# whether the owner has finished. core/endpointing.EotDecider asks it only in
# a real pause — at least SMART_TURN_MIN_SILENCE_S of RMS silence, a Silero
# silence run, and SMART_TURN_MIN_SPEECH_S of Silero speech so far — and a p
# of SMART_TURN_THRESHOLD or more ends the turn there. The 21 chunks stay the
# ceiling; Silero or Smart Turn missing, failing or too slow = today's turn.
#   SMART_TURN_MODE   'off'    no models are loaded;
#                     'shadow' the models run but nothing changes: each
#                              owner turn JARVIS accepts gets an
#                              [eot-shadow] line saying when Smart Turn
#                              WOULD have ended it (resumed=1: a sound
#                              above the RMS gate came after that point);
#                     'on'     Smart Turn ends turns.
#                     Env JARVIS_SMART_TURN, or user_settings.json.
#   SMART_TURN_MODEL  the ONNX file (smart-turn-v3.2-cpu.onnx), kept OUTSIDE
#                     the repo: never commit a model. Missing = Smart Turn
#                     latched off (one log line), i.e. 'rms' turns.
# Set via user_settings.json; applies on the next start.
SMART_TURN_MODE = (os.getenv("JARVIS_SMART_TURN", "shadow").strip().lower()
                   or "shadow")
SMART_TURN_THRESHOLD = 0.7
SMART_TURN_MIN_SILENCE_S = 0.256
SMART_TURN_MIN_SPEECH_S = 1.0
SMART_TURN_MODEL = r"C:\JARVIS-models\smart-turn-v3\smart-turn-v3.2-cpu.onnx"

# ─── Turn checker (core/turn_checker.py, 2026-10-02) ──────────────────────
# After every LLM turn (spoken or typed) the checker asks whether the reply
# failed in a way a retry on Claude would fix: it claimed an action that never
# ran ("Done, sir." and nothing happened), ran nothing for a clear command, or
# named an action that does not exist. Route, glance, barged and self-voiced
# turns are never checked.
#   TURN_CHECK_MODE   'off'    no check at all;
#                     'shadow' check only: a "[turn-check]" log line for a
#                              failed turn and one row per checked turn in
#                              data/turn_check.jsonl (kind, confidence and
#                              action names - never the words);
#                     'on'     a failed turn at or above the checker's bar is
#                              retried ONCE on Claude, after "One moment,
#                              sir.", when the cloud is allowed for chat
#                              (backend claude, a key, chat not routed local)
#                              and the model is not one Anthropic already
#                              answered not_found for this session (then its
#                              CLAUDE_MODEL_SUCCESSORS entry, or no retry).
#                     Any other value reads as 'shadow'.
#   TURN_CHECK_ESCALATE_MODEL  the Claude model that retry runs on.
# Set via user_settings.json; applies on the next start.
TURN_CHECK_MODE = "shadow"
TURN_CHECK_ESCALATE_MODEL = "claude-sonnet-5-5"

# ─── Deterministic fast paths (core/fast_paths.py + core/date_math.py) ──
# When True, relative-date questions ("what's the date tomorrow", "how many
# days until Christmas", "how long until Friday"), "what did I just ask you"
# and "what's my name" (from USER_NAME) are answered right before the LLM,
# from the clock, this conversation and the config: correct and instant where
# the local model guessed. Voice and typed turns alike; no processing filler.
# Anything they don't fully understand still goes to the LLM. Set via
# user_settings.json; applies on the next start.
FAST_PATHS_ENABLED = True

# ─── Instant actions (core/instant_actions.py, 2026-10-02) ──────────────
# A basic command — volume up / down / mute / unmute, pause / resume / next /
# previous track, lights on / off, pause the print — matched by anchored,
# precision-first rules. Never a question, a "can you ...?" ask, two commands
# in one sentence, a pronoun ("turn it up") or anything that needs a yes.
# INSTANT_ACTIONS_MODE:
#   "shadow" (default) the brain still answers every turn. The action that
#            WOULD have run and whether the brain ran the same one go to
#            data/instant_actions.jsonl (time + action names only, never the
#            owner's words). Score it: python tools/instant_actions_report.py
#   "on"     the action runs at once with a short spoken line; no LLM call.
#   "off"    nothing is matched or logged.
# INSTANT_ACTIONS_ALLOW: the registered action names it may run. It can only
# narrow core/instant_actions.RULE_ACTIONS (a name no rule produces is
# ignored). Set via user_settings.json; applies on the next start.
INSTANT_ACTIONS_MODE = "shadow"
INSTANT_ACTIONS_ALLOW = [
    "volume_up", "volume_down", "volume_mute", "volume_unmute",
    "pause_music", "resume_music", "next_song", "previous_song",
    "smart_home_control", "pause_print",
]

# ─── Teams unread-message nudger (skills/teams_nudge.py) ──────────────────
# The background loop screenshots the screen every 10 minutes and asks the
# vision model whether Teams shows an unread badge. Off by default
# (2026-09-29): it never worked reliably for the owner, it cost a full-screen
# vision call on the shared local model, and a mis-read once produced a
# garbage spoken nudge. The on-demand check_teams action still works when
# asked. Set true in user_settings.json to bring the loop back; applies on the
# next start.
TEAMS_NUDGE_ENABLED = False

# ─── Night quieting (core/night_quiet.py) ────────────────────────────────
# NIGHT_QUIET_ENABLED — the master switch for everything that changes how
# JARVIS sounds at night because of the CLOCK alone: the 'hushed_late' voice
# from 23:00 (gain 0.55), the late-night tone and mood from 22:00 (a quieter,
# slower voice and one-sentence replies), the time-only 'tired' read, the
# prompt's "a late hour means quietly stressed" rule, the "we've been at this
# a while" late-hour nudge, the 23:00-07:00 hold on his other proactive lines
# once you have been silent for 30 minutes, the softer wake greeting and its
# 01:00-04:59 "Still up, sir?", the 01:00-04:59 spoken remark about the hour,
# and night-owl mode switching itself on at 23:00. False = his voice, reply
# length and proactive speech at night are the same as in the daytime. What
# you ask for still works: "night owl on", or telling him you are tired. Set
# via user_settings.json; applies on the next start.
NIGHT_QUIET_ENABLED = True

# ─── Night-owl mode (skills/night_owl_mode.py) ───────────────────────────
# NIGHT_OWL_AUTO — at 23:00 JARVIS switches night-owl mode on by himself
# until 06:00: a voice about 15% quieter and about 5% slower, replies kept to
# one short sentence, no "thinking" filler clip, non-essential announcements
# (weather, news, banter, wellness, anticipation, screen-watch) held, and the
# overlay dimmed. False stops the AUTOMATIC switch-on only; "night owl on" /
# "night owl off" by voice still work. NIGHT_QUIET_ENABLED = False stops it
# too. Set via user_settings.json; applies on the next start.
NIGHT_OWL_AUTO = True


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


# ─── Double-clap trigger (skills/clap_trigger.py, core/clap_detector.py) ─
# CLAP_TRIGGER_ENABLED — when True, two sharp claps ~0.15-0.7 s apart with
#   nothing else loud around them run the clap routine. It listens through the
#   main loop's mic fan-out (add_record_tap) — never a second stream — so it
#   only hears while JARVIS is listening, never while he speaks (his own
#   playback is gated out too), never on staging. Off by default; "turn on the
#   clap trigger" / "clap trigger off" toggle it live and persist it.
# CLAP_TRIGGER_ACTION — what a double clap runs. Default "acknowledge" = just
#   "You rang, sir?" (2026-10-02 review: a mechanical key's click can pass for a
#   clap, so until you have heard it answer only your claps a false trigger
#   costs one line). "predictive_morning_setup" = the morning workspace setup
#   (Chrome + Apple Music, Teams, master volume ~30%); "morning_briefing" = the
#   briefing. Those are the ONLY routines a clap runs (an allow-list): any other
#   name is refused out loud. "Clap trigger runs the morning setup" by voice
#   sets it too.
# CLAP_TRIGGER_WAKE — "clap to wake": a double clap while asleep / in standby
#   wakes JARVIS and runs the routine. Off = claps are ignored while asleep.
#   Claps are ignored during the quiet hours (PHONE_PING_QUIET_START-END), in
#   focus mode, in game mode, and while music, a video or anything else is
#   playing on the speakers.
# CLAP_TRIGGER_COOLDOWN_S — minimum gap between two routines.
# CLAP_TRIGGER_MIN_PEAK — how loud a clap must be at the mic (peak sample,
#   0..1 full scale). Lower it if "clap trigger status" says your claps peak
#   below it; raise it if knocks across the room fire it.
CLAP_TRIGGER_ENABLED = False
CLAP_TRIGGER_ACTION = "acknowledge"
CLAP_TRIGGER_WAKE = False
CLAP_TRIGGER_COOLDOWN_S = 60.0
CLAP_TRIGGER_MIN_PEAK = 0.12


# ─── Phone pings (core/phone_ping.py, skills/phone_bridge.py) ──────────
# JARVIS texts your phone ONLY when something needs you, through the phone
# bridge (Telegram / ntfy / Pushover in .env). With no bridge configured every
# ping is a no-op and the boot log says so once; "how do I connect my phone"
# walks you through the @BotFather steps.
# PHONE_PING_ENABLED — the master switch for print / confirmation / robot
#   pings and the summary. On by default, but it does nothing until a backend
#   can send an unsolicited message (a Telegram token AND your
#   TELEGRAM_USER_ID, or an ntfy topic, or Pushover). "turn off phone pings" /
#   "turn on phone pings" flip it live and save it. Guard alerts are NOT under
#   it: they have their own switch, PHONE_PING_SECURITY.
# PHONE_PING_PRINT — a print finished, failed (not one you cancelled), or
#   paused with an error.
# PHONE_PING_CONFIRM — OFF by default: a confirmation YOU asked for and left
#   unanswered while away. A confirmation lapses after 45 s (nothing runs), so
#   this is a heads-up that it did not happen, not a question (the action's
#   NAME only, never its argument).
# PHONE_PING_SECURITY — a guard-mode alert. Its own switch: "phone pings off"
#   does not stop it. Critical: it ignores quiet hours, focus mode and the
#   hourly cap (it has its own ceiling of 20 an hour).
# PHONE_PING_ROBOT — a robot event a skill reports with
#   skill_utils["ping_phone"]("robot", ...).
# PHONE_PING_SUMMARY — a once-a-day digest at PHONE_PING_SUMMARY_TIME ("07:30"
#   = a morning summary, "22:00" = a nightly one). Off by default. While it is
#   on it also carries what quiet hours held back.
# PHONE_PING_MAX_PER_HOUR — ordinary pings in any rolling hour.
# PHONE_PING_QUIET_START / _END — "HH:MM", local time; ordinary pings are held
#   and sent as one message when quiet hours end. Equal values = no quiet hours.
#   Overnight mode ("goodnight", while its flag is set) counts as quiet hours
#   too. The double-clap trigger ignores claps in this window as well.
# PHONE_PING_AWAY_MIN — an ordinary ping waits until you have said nothing to
#   JARVIS — and, while he is awake and talking, typed or moved the mouse —
#   for this many minutes (he already told you out loud); 0 = ping even while
#   you are talking to him.
# PHONE_PING_CONFIRM_AFTER_MIN — how old an unanswered confirmation must be
#   before it may ping (it also needs you away, above).
PHONE_PING_ENABLED = True
PHONE_PING_PRINT = True
PHONE_PING_CONFIRM = False
PHONE_PING_SECURITY = True
PHONE_PING_ROBOT = True
PHONE_PING_SUMMARY = False
PHONE_PING_SUMMARY_TIME = "07:30"
PHONE_PING_MAX_PER_HOUR = 6
PHONE_PING_QUIET_START = "23:00"
PHONE_PING_QUIET_END = "07:00"
PHONE_PING_AWAY_MIN = 10.0
PHONE_PING_CONFIRM_AFTER_MIN = 2.0


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

# CAMERA_BACKEND — which OpenCV capture backend actually opens a camera:
# "msmf" (Media Foundation, the default since 2026-09-05) or "dshow"
# (DirectShow, the historical behaviour). Override for one run with the env var
# JARVIS_CAMERA_BACKEND.
#
# DirectShow LEAKS. Measured on this rig 2026-09-05, 25 open/read/release
# cycles per figure, each in a fresh process, 1280x720 requested:
#     eMeet C960   CAP_DSHOW  +4.58 OS threads  +479 handles  per cycle
#     eMeet C960   CAP_MSMF   -0.21 OS threads    +0.97       per cycle
#     USB 2.0 Cam  CAP_DSHOW  +4.55 OS threads  +479 handles  per cycle
#     USB 2.0 Cam  CAP_MSMF   -0.18 OS threads    +1.06       per cycle
# The DirectShow threads are owned by mfksproxy.dll and are never reclaimed —
# the same leak v2.0.101 gated in the ENUMERATION path, reappearing in the OPEN
# path where no gate can help. A DEAD index costs +103 handles under DirectShow
# and +0 under Media Foundation, so index sweeps pay it too.
#
# The indices below are still DIRECTSHOW indices: the two backends enumerate
# DIFFERENT lists (DirectShow 0=USB 2.0 Camera 1=Kinect 2=eMeet 3=OBS Virtual
# Camera; Media Foundation 0=Kinect 1=USB 2.0 Camera 2=eMeet, no OBS at all),
# so core/camera_backend.py TRANSLATES at open time — by device NAME where
# CAMERAS supplies one. Do NOT hand a raw CAMERAS index to CAP_MSMF; on this
# rig that repoints index 0 from the USB webcam to the Kinect.
CAMERA_BACKEND            = "msmf"

# Camera probe — if CAMERAS fails to open, sweep indices 0..MAX-1 and
# rewrite CAMERAS with whatever's actually plugged in. cv2.CAP_DSHOW
# retries internally for ~26s on a missing index, so each probe is
# capped by CAMERA_PROBE_TIMEOUT_SEC.
CAMERA_PROBE_ENABLED      = True
CAMERA_PROBE_MAX          = 12     # probe indices 0..MAX-1
CAMERA_PROBE_TIMEOUT_SEC  = 3.0    # per-index hard timeout

# THE CAMERA OPEN GATE (core/camera_gate.py, 2026-09-29). Every camera and
# Kinect open in JARVIS asks one gate first. Measured live 2026-09-29: the
# owner's chained USB hubs reset ~once a minute while JARVIS kept reopening
# cameras after failures, and not once while nothing opened a camera or after
# JARVIS stopped. All three are floats on purpose (an int default would make
# _apply_user_settings truncate a saved 12.5 to 12). Apply on the next start.
#
# CAMERA_REOPEN_MAX_BACKOFF_S — ceiling of the per-camera retry ladder after a
#   failed open or a read-failure recovery: 30 -> 60 -> 120 -> 300 -> 600 s.
#   The ladder resets only after 60 s of uninterrupted healthy frames.
#   0 disables the ladder (not recommended).
CAMERA_REOPEN_MAX_BACKOFF_S = 600.0
# USB_STORM_COOLDOWN_S — the circuit breaker. When >=2 cameras (or a camera and
#   an audio device) drop within ~10 s, or >=3 camera opens fail across >=2
#   devices within 60 s, JARVIS treats it as a USB bus event and opens NO
#   camera and NO Kinect for this long (doubling on a repeat within the hour,
#   capped at 60 min). Streams already running are left alone. 0 disables it.
USB_STORM_COOLDOWN_S        = 600.0
# CAMERA_STORM_PROBATION_S — (v2.0.132) for this long after a cool-down ends
#   (and for 60 s after any reopen while that storm chain is live), ONE camera
#   drop - gone from the device list, a read-failure burst, or an audio device
#   vanishing - re-trips the breaker at once with the doubled cool-down.
#   Measured 2026-09-29: a webcam reopened 18 s after the first cool-down reset
#   the hub twice and nothing re-tripped. 0 disables the probation.
CAMERA_STORM_PROBATION_S    = 180.0
# CAMERA_CULPRIT_WINDOW_S / CAMERA_CULPRIT_THRESHOLD — (v2.0.132) a USB bus
#   event that begins within CAMERA_CULPRIT_WINDOW_S of ONE device's stream
#   start is a strike against that device; CAMERA_CULPRIT_THRESHOLD strikes
#   within an hour QUARANTINE it for the rest of the session (JARVIS says so
#   once and never opens it on its own again; "use the left webcam again"
#   lifts it). Measured 2026-09-29: one webcam's stream starts reset the hub
#   8 of 10 times, its sibling on the same hub 0 of 10. Either one at 0
#   disables the quarantine. The threshold is a count, so it is an int.
CAMERA_CULPRIT_WINDOW_S     = 5.0
CAMERA_CULPRIT_THRESHOLD    = 2
# CAMERA_DIES_ON_OPEN_RETRY_S — (R11) a camera or Kinect whose stream dies
#   within 15 s of each of 3 opens in a row (it drops out a few seconds after
#   each start - when it leaves the bus, usually its power supply) is retried
#   only this often, doubling to at most an hour, instead of every 10
#   minutes. JARVIS says so once per session; a reopen that streams normally,
#   or "use the Kinect again", puts it back at once. Measured 2026-10-02: the
#   Kinect streams ~6 s, then drops off USB ~7.5 s after it is switched on,
#   whatever reads it and on any port. 0 disables it.
CAMERA_DIES_ON_OPEN_RETRY_S = 1800.0
# CAMERA_OPEN_MIN_GAP_S — two different JARVIS components (face tracker, boot
#   probe, self-diagnostic, side tiles, Kinect bridge) never open the same
#   device less than this far apart. 0 disables the gap.
CAMERA_OPEN_MIN_GAP_S       = 10.0

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
#   JARVIS from standby and a SWIPE dismisses/cancels (stop speech + clear the
#   pending confirmation). RAISE_HAND never confirms (2026-10-01: a stretch or
#   a guest's raised hand ran destructive actions) - it only reminds you that a
#   pending action needs a spoken "yes". Off by default; never runs in
#   staging/test. See
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
# KINECT_AIR_MOUSE_ENABLED — when True, "air-mouse": RAISE a hand above the
#   shoulder to take the cursor (the SMART-ENGAGE block right below is the single
#   source for the exact gate — don't restate it here); LOWERING the hand releases
#   it. Clicking is HAND-SPECIFIC: closing the LEFT hand presses the LEFT mouse
#   button, closing the RIGHT hand presses the RIGHT one, and holding either
#   closed while moving drags with that button. Either hand can click regardless
#   of which one drives the cursor (skills/kinect_air_mouse.py :: engage_decision
#   → _apply_decision → _mouse_button).
#   A background poller (~30 Hz) maps the driving hand's position within a
#   calibrated reach-box onto the WHOLE VIRTUAL DESKTOP — every monitor, incl.
#   any left of / above the primary, so target pixels may be negative
#   (_reach_box_for_virtual_desktop) — smoothed by the shared hand
#   stabiliser's One Euro filter (KINECT_HAND_FILTER_* below) to fight
#   jitter, and drives the cursor via win32api (pyautogui fallback). A glowing
#   JARVIS reticle (hud/jarvis_air_cursor.py) follows the cursor — cyan while
#   tracking an open hand, gold-locked on grab/drag. Off by default; never runs
#   in staging/test; a dead-man releases any held button the instant the hand
#   isn't tracked. See audio/kinect_bridge.get_hand_states() (grip) +
#   skills/kinect_air_mouse.py (wiring) + hud/jarvis_air_cursor.py (overlay).
KINECT_AIR_MOUSE_ENABLED = False
# AIR_MOUSE_LL_HOOK_ENABLED — install the low-level mouse/keyboard hook the
# air-mouse uses to tell real input from its own (skills/_air_mouse_yield.py).
# OFF: a Python LL hook puts every input event on the PC behind this process's
# GIL (~15.6 ms per event measured under JARVIS's load, 311 ms per 20-event
# burst — games included). The GetLastInputInfo fallback does the job.
AIR_MOUSE_LL_HOOK_ENABLED = False
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
# AIR_MOUSE_ARM_TIMEOUT_SEC — how long the ARMED (relaxed) window lasts, refreshed
#   for as long as the owner is actually driving the cursor. ARMED used to LATCH
#   for the whole session: after one "take the cursor" every later raised hand
#   could re-grab it on height alone, with no open-palm / facing / stillness /
#   reach test. It now lapses back to the strict PASSIVE gate once he stops using
#   it. ~120 s. 0 restores the old latch-forever behaviour.
AIR_MOUSE_ARM_TIMEOUT_SEC = 120.0
# ─── PASSIVE ACQUIRE REACH CONJUNCT (2026-09-04) ──────────────────────────────
# The PASSIVE gate's other tests (open palm, facing, still) are all satisfied MORE
# easily by someone sitting motionless than by someone reaching, so a raised hand
# above the shoulder line was effectively the ONLY requirement — and any relaxed
# posture that parks a hand high (hand on the head, arm on an armrest or the back
# of the couch, a stretch) took the cursor. These two bars add the one thing a
# resting arm does NOT do: REACH. Fresh PASSIVE acquisition only — never a
# stay-engaged veto (no stutter), never in ARMED mode.
# AIR_MOUSE_ENGAGE_REACH_RATIO — forward reach ÷ body scale (shoulder span). Used
#   whenever a body scale is measurable, because an absolute metre bar is NOT
#   position-independent (the same gesture measures less forward reach the farther
#   away the owner sits). RAISE if a resting posture still takes the cursor; LOWER
#   if a genuine reach fails to; 0 disables the forward half.
AIR_MOUSE_ENGAGE_REACH_RATIO = 0.50
# AIR_MOUSE_ENGAGE_REACH_M — absolute forward-reach fallback in metres, used only
#   when no shoulder span / torso height is measurable this frame.
AIR_MOUSE_ENGAGE_REACH_M = 0.20
# AIR_MOUSE_ENGAGE_STRAIGHT — arm straightness (shoulder→hand chord ÷ arm length,
#   0..1) required to engage. A pure ratio of the owner's own arm, so it is immune
#   to distance AND to torso lean — unlike forward reach, whose spine reference
#   shifts when he sits back. A reach extends the elbow; a resting arm is folded.
#   0 disables the elbow half.
AIR_MOUSE_ENGAGE_STRAIGHT = 0.85
# AIR_MOUSE_GRIP_CLOSE_FRAMES — LEGACY, used only when the bridge has no shared
#   hand stabiliser; the live press rule is KINECT_GRIP_CLOSE_SEC +
#   KINECT_GRIP_CLOSE_MIN_VOTES below (High-confidence REAL frames, not polls).
#   Consecutive polls a CLOSED hand must be seen
#   before it presses a button. Deliberately STRICTER than the open/release
#   debounce: a spurious release only drops a drag, a spurious press is a click the
#   owner never made (one closed his browser tabs). ~4 frames ≈ 133 ms at 30 Hz —
#   past the correlated 2-3-frame misreads the Kinect hand classifier produces at
#   range, which a 2-frame bar did not cover. NB "lasso" also votes CLOSED.
AIR_MOUSE_GRIP_CLOSE_FRAMES = 4
# AIR_MOUSE_OFFHAND_CLICK_MIN_LIFT_M — clicks are per hand (left hand = left
#   button, right hand = right button) whichever hand drives the cursor, but the
#   hand that is NOT driving may only press while it is held up at least this
#   high relative to the shoulder line (metres; -0.10 = upper-chest height).
#   A fist resting on the desk (~-0.30) never clicks, and a held button is let
#   go the moment that hand drops below it. -1.0 = click from anywhere (the
#   pre-2026-10-04 behaviour); 1.0 = only the driving hand clicks. -0.10 m.
AIR_MOUSE_OFFHAND_CLICK_MIN_LIFT_M = -0.10
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
# ─── KINECT HAND STABILISER (2026-10-04, "hand tracking is unstable") ─────────
# One shared stabiliser (audio/kinect_stabilizer.py) runs ONCE per real Kinect
# body frame inside the bridge pump; the air-mouse, two-hand, gestures and
# pointing all read its snapshot (audio.kinect_bridge.get_tracked_frame), so
# they agree on the owner body, the active hand, raised / grip state and
# two-hand mode. Every value below is also the module's own default
# (kinect_stabilizer.DEFAULTS - a test pins the two equal). Override any of them
# in data/user_settings.json; they are read live every frame.
# Measured basis (2026-10-04 desk recording, scratchpad kinect_tracking/): hand
# joint Tracked only 29-35% of frames vs the WRIST 50-94%; still-hand jitter
# 3.6-4.8 mm RMS (p90 24-36 mm); single-frame steps up to 186 mm; the grip
# classifier LOW-confidence on 80% of left-hand frames at rest.
# KINECT_HAND_FILTER_MIN_CUTOFF_HZ — One Euro filter cutoff for a hand at REST.
#   Lower = steadier cursor at rest, more lag on very slow moves. 0.7 Hz.
KINECT_HAND_FILTER_MIN_CUTOFF_HZ = 0.7
# KINECT_HAND_FILTER_BETA — how fast the cutoff opens with hand SPEED (Hz per
#   m/s). Higher = less lag when moving, more jitter on slow moves. 12.0 keeps
#   the added lag under 40 ms from ~0.25 m/s up.
KINECT_HAND_FILTER_BETA = 12.0
# KINECT_HAND_FILTER_D_CUTOFF_HZ — cutoff of the speed estimate the filter
#   adapts on. 1.0 Hz (the One Euro paper's default).
KINECT_HAND_FILTER_D_CUTOFF_HZ = 1.0
# KINECT_HAND_JUMP_REJECT_M — a hand that jumps further than this in ONE frame
#   (scaled by frames elapsed; ~7.5 m/s, faster than a real hand) is a glitch:
#   the frame is dropped. 0.25 m.
KINECT_HAND_JUMP_REJECT_M = 0.25
# KINECT_HAND_JUMP_REJECT_FRAMES — at most this many consecutive jump frames are
#   dropped; if the hand is still "there" after that it is real and the filter
#   re-seeds at the new place. 2.
KINECT_HAND_JUMP_REJECT_FRAMES = 2
# KINECT_HAND_LOSS_GRACE_SEC — when the hand (and wrist) stop being measured,
#   HOLD the last good position / lift / grip this long before reporting the
#   hand as gone (which releases the cursor). One Inferred frame used to drop
#   the lift to None and release instantly. 0.30 s.
KINECT_HAND_LOSS_GRACE_SEC = 0.30
# KINECT_HAND_OFFSET_CUTOFF_HZ — the hand-wrist offset is low-pass filtered at
#   this cutoff, and a Tracked WRIST + that offset is the PREFERRED hand position
#   (the wrist is steadier, and closing the hand moves the hand joint but not
#   the wrist). The last offset is kept however long the hand joint stays
#   un-Tracked, and re-learned smoothly - never a step. Measured on the desk
#   recording: still-hand jitter p90 22 mm (old cursor path) -> 10 mm. Higher =
#   follows wrist-only flexion faster, with more hand-joint noise. 1.0 Hz.
KINECT_HAND_OFFSET_CUTOFF_HZ = 1.0
# KINECT_GRIP_REQUIRE_HIGH_CONFIDENCE — a closed hand PRESSES a button only on
#   frames where the Kinect's own classifier is HIGH confidence and the hand
#   joint is Tracked; a LOW-confidence fist is "no vote" for a press. (Opening
#   the hand RELEASES at any confidence.) Set False only if clicks stop
#   registering at all (the SDK then never reports High for your hands).
KINECT_GRIP_REQUIRE_HIGH_CONFIDENCE = True
# KINECT_GRIP_CLOSE_SEC — a closed hand must be seen (High confidence, real
#   frames, no open vote in between) across at least this span... 0.09 s.
KINECT_GRIP_CLOSE_SEC = 0.09
# KINECT_GRIP_CLOSE_MIN_VOTES — ...on at least this many REAL frames before it
#   PRESSES a button. 4 frames = a closed hand seen for 133 ms; a flicker of 3
#   frames (100 ms) or less can never click, whatever its timestamps say. 4.
KINECT_GRIP_CLOSE_MIN_VOTES = 4
# KINECT_GRIP_OPEN_SEC — the same for opening (RELEASE): span... 0.03 s.
KINECT_GRIP_OPEN_SEC = 0.03
# KINECT_GRIP_OPEN_MIN_VOTES — ...and frames. Looser than the press: a spurious
#   release only drops a drag, a spurious press is a click you never made. 2.
KINECT_GRIP_OPEN_MIN_VOTES = 2
# KINECT_GRIP_VOTE_GAP_SEC — a run of grip votes is broken when no vote arrives
#   for longer than this (the run must start again). 0.10 s.
KINECT_GRIP_VOTE_GAP_SEC = 0.10
# KINECT_GRIP_LASSO_AS — what a LASSO (two-finger point) reading counts as for
#   a click: "none" (no vote - the default: a pointing hand is not a click),
#   "closed" (the pre-2026-10-04 behaviour) or "open". Whatever the setting, a
#   pointing hand is never an OPEN PALM for AIR_MOUSE_REQUIRE_OPEN_PALM.
KINECT_GRIP_LASSO_AS = "none"
# KINECT_GRIP_CLOSED_HOLD_MAX_SEC — a held fist (button down) with NO closed
#   reading at all for this long (the SDK says Unknown / the hand joint is only
#   Inferred) lets the button go, so a drag can't outlive the evidence for it.
#   0 = hold until the hand is seen open or lost. 1.0 s.
KINECT_GRIP_CLOSED_HOLD_MAX_SEC = 1.0
# KINECT_LIFT_UP_MARGIN — a hand counts as RAISED once its height above the
#   shoulder line passes this (metres). Same key the 'calibrate air mouse'
#   action persists, so a calibration moves the shared gate and the air-mouse
#   together. 0.07 m.
KINECT_LIFT_UP_MARGIN = 0.07
# KINECT_LIFT_DOWN_MARGIN — ...and stays raised until it drops below this
#   (hysteresis: DOWN < UP, so a hand at the line can't flap). -0.10 m.
KINECT_LIFT_DOWN_MARGIN = -0.10
# KINECT_RAISE_ENTER_SEC — the hand must stay above UP this long to count as
#   raised (a one-frame height spike doesn't). 0.10 s.
KINECT_RAISE_ENTER_SEC = 0.10
# KINECT_RAISE_EXIT_SEC — ...and below DOWN this long to count as lowered. 0.10 s.
KINECT_RAISE_EXIT_SEC = 0.10
# KINECT_ACTIVE_HAND_SWITCH_LEAD_M — with both hands raised, the cursor moves to
#   the other hand only when that hand is this much HIGHER... 0.15 m.
KINECT_ACTIVE_HAND_SWITCH_LEAD_M = 0.15
# KINECT_ACTIVE_HAND_SWITCH_SEC — ...for this long (or at once if you lower the
#   driving hand). 0.40 s.
KINECT_ACTIVE_HAND_SWITCH_SEC = 0.40
# KINECT_TWO_HAND_ENTER_SEC — both hands must stay raised this long before
#   two-hand (pinch-to-resize) mode engages and the single-hand cursor stands
#   down. 0.25 s.
KINECT_TWO_HAND_ENTER_SEC = 0.25
# KINECT_TWO_HAND_EXIT_SEC — two-hand mode ends only after "not both raised"
#   has lasted this long. 0.20 s.
KINECT_TWO_HAND_EXIT_SEC = 0.20
# KINECT_TWO_HAND_REARM_SEC — after two-hand mode ends it can't re-engage for
#   this long, so engage edges are >= EXIT + REARM + ENTER (1.05 s) apart and it
#   cannot flap within a second. 0.60 s.
KINECT_TWO_HAND_REARM_SEC = 0.60
# KINECT_TWO_HAND_ENTER_ABOVE_M — to ENTER two-hand mode both hands must be at
#   least this far ABOVE the raise line (KINECT_LIFT_UP_MARGIN). Raise it (e.g.
#   0.05) if two-hand mode takes over while your second hand only hovers near
#   your chin and you wanted the cursor. 0.0 m.
KINECT_TWO_HAND_ENTER_ABOVE_M = 0.0
# KINECT_TWO_HAND_EXIT_BELOW_M — two-hand mode ENDS once either hand drops this
#   far below the raise line (for KINECT_TWO_HAND_EXIT_SEC). Its own narrow band,
#   not the single-hand stay line: a second hand resting at chin / chest height
#   gives the cursor back. 0.04 m.
KINECT_TWO_HAND_EXIT_BELOW_M = 0.04
# KINECT_OWNER_LOSS_GRACE_SEC — if the tracked owner body vanishes for a frame,
#   hold it this long before treating it as gone. 0.30 s.
KINECT_OWNER_LOSS_GRACE_SEC = 0.30
# KINECT_OWNER_SWITCH_NEARER_M — another body takes over as the owner only when
#   it is at least this much nearer the sensor... 0.25 m.
KINECT_OWNER_SWITCH_NEARER_M = 0.25
# KINECT_OWNER_SWITCH_SEC — ...for this long. 1.0 s.
KINECT_OWNER_SWITCH_SEC = 1.0
# KINECT_OWNER_CLAIM_SEC — a body at least as near the sensor as the current
#   owner that holds a hand RAISED this long, while the owner has no hand
#   raised, becomes the owner: you can always take the cursor back by raising a
#   hand, even if someone farther away was picked first. 0.30 s.
KINECT_OWNER_CLAIM_SEC = 0.30
# KINECT_SNAPSHOT_STALE_SEC — with NO new body frame for this long (the pump
#   starved, the sensor stopped) the hand state reads as not tracked and the
#   cursor / any held button is let go. A shorter stall just holds. 0.60 s.
KINECT_SNAPSHOT_STALE_SEC = 0.60
# KINECT_GESTURE_RELEASE_MUTE_SEC — gestures (wave / swipe / raise) stay muted
#   this long after the air-mouse or two-hand mode lets go, so the arm coming
#   down can't fire a SWIPE (which cancels speech and a pending confirmation).
#   0.5 s.
KINECT_GESTURE_RELEASE_MUTE_SEC = 0.5
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

# ─── Brain glow (core/brain_glow.py) ─────────────────────────────────────
# BRAIN_GLOW_ENABLED — a glowing ring around the HUD orb / arc reactor takes
#   the colour of the brain that is answering: local model = blue, Claude
#   Sonnet = gold, Opus = violet, Haiku = teal, Fable = rose, any other cloud
#   model = silver. It changes the moment you switch brains (set_model /
#   set_brain / switch_llm) and per turn when a turn is really answered by the
#   other brain (a local turn the cloud had to answer, a cloud turn that fell
#   back to local). The halo and core keep the state colours (listening /
#   thinking / speaking), so Sonnet's gold never reads as "thinking"; asleep /
#   in standby there is no ring. Off = the HUDs look exactly as before. Read
#   live on every publish; one hud_state.json write per brain CHANGE, never
#   per turn.
# BRAIN_GLOW_LABEL_S — seconds the brain's name shows under the reactor after
#   a change (0 = colour only, never a label).
# BRAIN_GLOW_COLORS — per-tier colour overrides, e.g. {"opus": "#FF3B3B"}.
#   Tiers: local, haiku, sonnet, opus, fable, cloud. Merged over the defaults
#   in core/brain_glow.DEFAULT_COLORS; a bad hex or unknown tier is ignored.
#   Not in the Settings window (no colour picker) — set it in
#   data/user_settings.json.
BRAIN_GLOW_ENABLED = True
BRAIN_GLOW_LABEL_S = 4.0
BRAIN_GLOW_COLORS = {}

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
# DASHBOARD_SHOW_TRANSCRIPTS — the dashboard's "What JARVIS did" timeline shows
# what was SAID on each turn (the owner's words, JARVIS's reply) only when this
# is True AND the browser is on this PC (a loopback request). A LAN client never
# gets the text, token or not. Off: the timeline shows times, sources, actions
# and latencies only. Read at boot (applies on the next start).
DASHBOARD_SHOW_TRANSCRIPTS = False

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
# the power state.
#
# CORRECTED 2026-09-05. The sentence that stood here until now said "Windows
# flips the headset's audio ENDPOINT Active<->NotPresent instead; this watcher
# polls that". That is FALSE and was never measured. Measured 2026-09-04 with
# the CORSAIR VOID ELITE POWERED OFF, Windows reported BOTH of its endpoints
# Status=OK / Active, because the endpoint belongs to the DONGLE. The watcher
# reads the dongle's Corsair vendor HID report (audio/void_link.py) instead,
# which is three-valued (on / off / genuinely unknown), and unknown HOLDS.
#
#   PLAYBACK  headset ON  -> default = the headset (remembers the prior default)
#             headset OFF -> prior default, else AUDIO_AUTOSWITCH_FALLBACK
#   RECORDING (AUDIO_AUTOSWITCH_MIC only)
#             headset ON  -> default mic = the headset's microphone
#             headset OFF -> remembered mic, else AUDIO_AUTOSWITCH_MIC_FALLBACK
#
# Opt-in. HEADSET/FALLBACK are case-insensitive substrings of the Windows
# device friendly name (see `python -m audio.audio_switch --list`).
AUDIO_AUTOSWITCH_ENABLED  = os.getenv("JARVIS_AUDIO_AUTOSWITCH", "").lower() in ("1", "true", "yes", "on")
AUDIO_AUTOSWITCH_HEADSET  = os.getenv("JARVIS_AUDIO_HEADSET", "")    # e.g. "CORSAIR VOID ELITE"
AUDIO_AUTOSWITCH_FALLBACK = os.getenv("JARVIS_AUDIO_FALLBACK", "")   # e.g. "Realtek USB2.0 Audio"


def _env_float(name: str, default: float) -> float:
    """float(os.getenv(name)), or `default` when it is unset or blank. A value
    that is not a finite number also gets `default`, plus ONE warning line on
    stderr: this runs at import, and the monolith imports core.config at module
    top level, so a bare float("abc") here stopped JARVIS from booting at all
    (audit A86)."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("not finite")
        return value
    except (TypeError, ValueError):
        try:
            import sys
            print(f"[config] WARNING: {name}={raw[:40]!r} is not a number; "
                  f"using the default {default}", file=sys.stderr, flush=True)
        except Exception:
            pass
        return default


AUDIO_AUTOSWITCH_POLL_S   = _env_float("JARVIS_AUDIO_POLL_S", 3.0)

# The INPUT half — make the MICROPHONE follow the headset's power too.
# Separate from AUDIO_AUTOSWITCH_ENABLED and OFF by default: it writes the
# default RECORDING endpoint, and getting that wrong is how JARVIS goes deaf
# (measured 2026-09-05 00:48-00:52 — the output moved to the speakers, the
# input stayed on the powered-off headset, and the VAD read peak RMS 0.0000).
AUDIO_AUTOSWITCH_MIC = os.getenv("JARVIS_AUDIO_MIC_FOLLOW", "").lower() in ("1", "true", "yes", "on")

# The desk microphone to fall back to when the headset powers off. It needs its
# OWN setting and must NOT borrow AUDIO_AUTOSWITCH_FALLBACK: measured
# 2026-09-05, that setting's value on this machine ("Realtek USB2.0 Audio") has
# ZERO Active RECORDING endpoints — Line In and Microphone both Unplugged,
# Internal AUX Jack Disabled — so reusing it would resolve to nothing and
# silently no-op, which is the exact failure shape (a device of the wrong
# direction sitting in a fallback slot, saying nothing) that this module was
# rewritten to eliminate. Blank = the input half has nothing to fall back to
# and will say so loudly rather than move the microphone somewhere unverified.
# Check a candidate with `python -m audio.audio_switch --list-mics`.
AUDIO_AUTOSWITCH_MIC_FALLBACK = os.getenv("JARVIS_AUDIO_MIC_FALLBACK", "")  # e.g. "Blue Snowball"

# How long the SELECTED headset microphone may produce literal digital silence,
# while the capture loop is demonstrably running, before the watcher moves off
# it. This is the ON side's only signal test and the only place in the audio
# auto-switch that measures HEARING rather than SELECTION.
#
# WHY IT EXISTS. The ON side fired once, on the power-up transition, and then
# never looked again: `if was_on is True: return None` was the whole
# steady-state branch. So a headset that is powered ON but hearing nothing —
# boom mic flipped up (its own hardware mute), the endpoint muted in Windows,
# the owner in another room — got the default recording endpoint moved onto it
# and kept it, for hours. The OFF rescue cannot help there, because the headset
# MEASURES powered on, and nothing consulted the microphone's actual signal.
#
# 60 s, not 30. core.audio_processor already WARNS at MIC_SILENT_WARN_SECONDS
# = 30 s; this MOVES a device, so it wants a longer, more confident window and
# deliberately sits behind that warning rather than racing it.
# <= 0 switches the watchdog off entirely (the ON side then behaves as it did
# before 2026-09-05, i.e. fire-once-and-never-re-check).
AUDIO_AUTOSWITCH_MIC_SILENT_S = _env_float("JARVIS_AUDIO_MIC_SILENT_S", 60.0)


# ─── Audio-device flap damping (bobert_companion + core/audio_flap.py) ─
# 2026-09-29: a desk mic's Windows endpoint went Active <-> NotPresent every
# ~20-40 s (USB still connected), Windows bounced the default recording device
# between it and a powered-off headset, and JARVIS spoke 17 audio-device
# sentences in ~10 minutes to an empty room. Every audio-device sentence
# ("Switched to ...", "I may not be able to hear you ...", its recovery line,
# speaker switches) now goes through one governor:
#   * a device that changes AUDIO_FLAP_THRESHOLD times inside
#     AUDIO_FLAP_WINDOW_S is FLAPPING: ONE plain sentence, then quiet until it
#     has gone twice that window (10 min at the defaults) without a change,
#     which is also said once. AUDIO_FLAP_THRESHOLD below 2 turns it off.
#   * at most one audio-device sentence per AUDIO_ANNOUNCE_MIN_GAP_S; a newer
#     one replaces a held or still-queued older one. 0 turns the gap off.
#   * the capture device follows a moved Windows default only once the new
#     default has held for AUDIO_REPICK_STABLE_S, unless the current device is
#     gone (no longer an Active endpoint). 0 = follow on the first pass.
# Floats are float literals (an int default would make _apply_user_settings
# truncate a saved 7.5 to 7). Changes apply on the next start.
AUDIO_FLAP_WINDOW_S      = 300.0   # s: window the changes are counted in
AUDIO_FLAP_THRESHOLD     = 3       # changes inside the window = flapping
AUDIO_ANNOUNCE_MIN_GAP_S = 60.0    # s: at most one audio-device sentence per
AUDIO_REPICK_STABLE_S    = 8.0     # s a new default must hold before JARVIS follows it


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

# APPLE_MUSIC_PLAYLIST_LINKS — direct links for "play my X playlist", as
# {"playlist name": "https://music.apple.com/..."}: the playlist page's address
# (https://music.apple.com/library/playlist/p.XXXX) or its Share > Copy Link.
# A named playlist opens straight on its own page, so nothing has to find its
# tile on screen first. Names match case- and apostrophe-insensitively; only
# https://music.apple.com/ links are used. Empty = the Library > Playlists
# route (keyboard first, vision last).
APPLE_MUSIC_PLAYLIST_LINKS: dict = {}


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
# RAG_EXCLUDE_GLOBS — what file search must never read. fnmatch globs, matched
# case-insensitively, with / and \ treated alike. Two kinds:
#   * a pattern WITH a slash is matched against the whole path
#     ("*/node_modules/*" skips every node_modules folder);
#   * a pattern WITHOUT a slash is matched against the file's NAME and the
#     name of every folder between the watched root and the file, so
#     "*password*" skips "Router Passwords.txt" AND "Passwords/bank.md".
#     Folders at or above the watched root (your user folder, the root
#     itself) are never tested, so a root under ".../Compass/" still works.
# The first block is the indexer's structural skips. The second is the
# secret-shaped names: passwords, secrets, credentials, tokens, API and
# private keys, .env / id_rsa / .pem / .key / .pfx / .kdbx files, wallets,
# recovery and backup codes, BitLocker, 2FA, and device-inventory exports.
# The third is every .csv / .tsv export: they often hold credential or device
# dumps. "*pass*" also catches names like "compass" or "passport" on purpose:
# for secrets a missed search beats a leaked one.
# A file that is already indexed and later matches a pattern is DROPPED from
# the index on the next scan (the file itself is never touched), and search
# refuses its hits straight away, before that scan has finished.
# OVERRIDE = REPLACE: a RAG_EXCLUDE_GLOBS saved in data/user_settings.json
# REPLACES this whole list; it is NOT merged with it. To add a pattern, copy
# the full list into the settings file and add to the copy — a saved list
# without the secret patterns lets secret files be indexed again.
RAG_EXCLUDE_GLOBS = [
    # structural
    "*/.git/*", "*/node_modules/*", "*/__pycache__/*", "*/.venv/*",
    "*/venv/*", "*/dist/*", "*/build/*", "*/.cache/*", "*/.next/*",
    "*/Library/Caches/*", "*/AppData/Local/*", "*/AppData/Roaming/*",
    "*.tmp", "*.lock", "*.cache",
    # secret-shaped names
    "*pass*", "*password*", "*passwd*", "*secret*", "*credential*",
    "*creds*", "*token*", "*apikey*", "*api?key*", "*private?key*",
    "*privatekey*", "*.env*", "*id_rsa*", "*.pem*", "*.key*", "*.pfx*",
    "*.kdbx*", "*wallet*", "*recovery*", "*bitlocker*", "*2fa*",
    "*backup?codes*", "*backupcodes*", "*level?devices*",
    # spreadsheet exports
    "*.csv", "*.tsv",
]


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
# core), which falls back to CPU/int8 if the GPU load fails. The card is the
# listen card (LISTEN_GPU), never cuda:0 = the brain's 3090 (2026-10-04).
STANDBY_WHISPER_PREFER_GPU        = False


# ─── User settings overrides (data/user_settings.json) ─────────────────
# The tray Settings GUI (tools/settings_window.py) writes data/user_settings.json
# (gitignored). Apply those overrides over the defaults above so a saved setting
# takes effect on the next start — and because bobert_companion.py does
# `from core.config import *` AFTER this module finishes importing, every
# consumer (the monolith, core.voice_pipeline, the skills) sees the overridden
# value with no extra wiring.
#
# Secrets are env/.env ONLY (see the BAMBU_* constants: "never commit"). The
# apply-loop below blindly overrides any config constant named in the settings
# file, so a credential written there — as the printer-reconnect flow once did —
# would defeat the env contract AND persist a secret into a settings file. These
# keys are therefore never sourced from user_settings.json. 2026-07-15.
_ENV_ONLY_KEYS = frozenset({"BAMBU_PRINTER_IP", "BAMBU_ACCESS_CODE", "BAMBU_SERIAL"})


# ─── Game low-power mode (skills/game_mode.py) ─────────────────────────────
# JARVIS gets out of the way while a game runs: the local brain is repointed at
# a SMALLER tag (in process only — nothing here is persisted, so any restart
# undoes the whole mode with zero cleanup code), the big model is unloaded, and
# the camera/gesture/diagnostic luxuries pause. It never gates the LLM: there is
# no cloud on this box (AI_BACKEND=ollama, MODEL_ROUTING all-local), so
# suppressing local chat would MUTE JARVIS, not reroute him.
#
# These constants MUST be declared here even though the skill owns the logic:
# _apply_user_settings() below skips any key that is `not in globals()`, so an
# undeclared setting is silently dropped and the owner's saved value never
# reaches the runtime. That is this project's defining bug class.
#
# OFF by default — the mode is enabled live, deliberately, not by shipping it on.
GAME_MODE_ENABLED = False
# EXACT executable basenames, lower-case. NOT substrings: measured live
# 2026-09-04 there are four Fortnite processes at once (FortniteBootstrapper,
# FortniteLauncher, FortniteClient-Win64-Shipping_EAC_EOS, and the real
# FortniteClient-Win64-Shipping), and a substring test matches the EAC wrapper
# too. Ships with exactly one entry because one is all that has been measured;
# "JARVIS, treat this as a game" adds more.
GAME_MODE_PROCESS_HINTS = ["fortniteclient-win64-shipping.exe"]
# The brain to run while gaming. gemma4:12b — 7.6 GB on disk, ~9 GB calibrated
# @16k, MULTIMODAL so see_screen keeps working. Do NOT "escalate" to
# gemma4:latest: core/vram_budget.py calls it 4 GB, but live `ollama list` on
# 2026-09-04 measures the installed blob at 9.6 GB — BIGGER than gemma4:12b.
# game_mode._pick_game_brain() re-checks this against live /api/tags and refuses
# any candidate that is not measurably smaller than the tag it replaces.
GAME_MODE_BRAIN = "gemma4:12b"
GAME_MODE_POLL_SECONDS = 5.0
# The game must hold the FOREGROUND continuously for this long before the mode
# engages — entry fails closed, so a game merely launching behind his work never
# trips it. Exit is owned by PROCESS LIFETIME, so alt-tab never exits.
GAME_MODE_ENTER_DWELL_SECONDS = 20.0
GAME_MODE_REQUIRE_FOREGROUND = True
# The game pid must be gone this long before restoring — a crash-and-relaunch
# must not thrash two big model loads back to back.
GAME_MODE_EXIT_GRACE_SECONDS = 45.0
# Deadman ceiling (L3). Past this with the game gone, the watcher force-exits
# regardless of internal state; past 2x it exits unconditionally.
GAME_MODE_MAX_SECONDS = 43200.0          # 12 h
# How long to wait after the unload before taking the 'after' sample.
GAME_MODE_VERIFY_DELAY_SECONDS = 8.0
# Below this MEASURED VRAM delta the mode reports FAILURE and shows a failure
# state. A mode that claims memory it did not free is the defect this whole
# feature is written against.
GAME_MODE_MIN_VRAM_DELTA_MB = 3000
# How long Ollama keeps the local brain loaded after a request (2026-10-02).
# Every local chat / warm / re-prime request sends this same value (a request
# without one resets the resident timer). "20m" = today's behaviour; -1 keeps
# it loaded until game mode or a restart unloads it - the owner chose that on
# 2026-10-02 after a 10 s reload on the first command after a 20 min break.
# Any Ollama duration string ("2h") or a number of seconds also works.
LOCAL_KEEP_ALIVE = "20m"
# Re-ping the small model this often so it does not evict on Ollama's clock.
# `keep_alive: "20m"` is HARDCODED in the chat payload (bobert_companion.py:
# 11148), so a long keep_alive cannot be pinned from outside — but the refresh
# only ever fires when the model is ALREADY RESIDENT, so it can never itself
# cause a load. 0 disables it.
GAME_MODE_KEEP_WARM_SECONDS = 600.0
# Say nothing on entry by default — he is mid-match.
GAME_MODE_ANNOUNCE = False


# ── settings-window fixes 2026-09-30: an unreadable file is LOUD ──────────
# A user_settings.json that exists but can't be applied used to be skipped in
# silence, so JARVIS ran on the built-in defaults with nothing said. Now it is
# printed at import (console) and kept here for bobert_companion.setup_logging
# to repeat into the session log, which doesn't exist yet at import time.
_USER_SETTINGS_ERROR = None


def _report_user_settings_failure(path: str, why: str) -> None:
    global _USER_SETTINGS_ERROR
    _USER_SETTINGS_ERROR = (
        f"{path} could not be applied ({why}) — EVERY saved setting is being "
        f"ignored and JARVIS is running on the built-in defaults. Fix the file "
        f"(the Settings window shows where) and restart.")
    try:
        import sys
        print(f"[config] WARNING: {_USER_SETTINGS_ERROR}", file=sys.stderr,
              flush=True)
    except Exception:
        pass


# ── Safety lists: a saved file may ADD entries, never remove shipped ones ──
# Audit P3-2, 2026-10-01. The list branch below REPLACES a list wholesale, so a
# saved "CONFIRM_KEYWORDS": [] (a bad hand edit, a half-written file) made
# bobert_companion._needs_confirmation return False for EVERY purchase, delete
# and format — with nothing said. The shipped entries, captured HERE before the
# apply at the bottom of this file, are a floor: a saved list is unioned onto
# them, and a save that leaves one out (or isn't a list at all) is reported on
# stderr and kept in _SAFETY_SETTINGS_WARNINGS for setup_logging to repeat into
# the session log. Underscore-prefixed, so the apply loop can't override the
# floor itself. SCREENSHOT_PRIVACY_BLOCKLIST ships empty (opt-in), so its floor
# is empty today; the entries the owner adds still apply unchanged.
_SAFETY_LIST_BASELINE = {
    "CONFIRM_KEYWORDS": tuple(CONFIRM_KEYWORDS),
    "SCREENSHOT_PRIVACY_BLOCKLIST": tuple(SCREENSHOT_PRIVACY_BLOCKLIST),
}
_SAFETY_SETTINGS_WARNINGS: list = []


def _warn_safety_setting(msg: str) -> None:
    _SAFETY_SETTINGS_WARNINGS.append(msg)
    try:
        import sys
        print(f"[config] WARNING: {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _merge_safety_list(key: str, val, cur):
    """The value `key` gets from a saved `val`: the shipped floor plus every
    saved non-blank string not already in it (case-insensitive). A non-list
    `val` keeps `cur`. Either way a save that would shrink the list is
    reported, naming what it left out."""
    base = list(_SAFETY_LIST_BASELINE[key])
    if not isinstance(val, (list, tuple)):
        _warn_safety_setting(
            f"{key} in user_settings.json is a {type(val).__name__}, not a "
            f"list — ignored; the built-in safety list stays in force.")
        return cur
    saved = [v for v in val if isinstance(v, str) and v.strip()]
    seen = {b.strip().lower() for b in base}
    merged = list(base)
    for v in saved:
        if v.strip().lower() not in seen:
            seen.add(v.strip().lower())
            merged.append(v)
    wanted = {v.strip().lower() for v in saved}
    left_out = [b for b in base if b.strip().lower() not in wanted]
    if left_out:
        _warn_safety_setting(
            f"{key} in user_settings.json leaves out {', '.join(left_out)} — "
            f"built-in safety entries can't be removed, so they stay on.")
    return type(cur)(merged) if isinstance(cur, (list, tuple)) else merged


# Safe + best-effort (the second import-time I/O in this file, after
# RAG_INDEX_PATHS): we override ONLY a constant that already exists here — so the
# GUI's schema, which is curated FROM this file, is the allow-list — coerce to
# the existing constant's type, and leave the default on any error. A missing
# file (fresh install, before the GUI ever ran) is a silent no-op; a file that
# exists but can't be read is reported (above).
def _apply_user_settings() -> None:
    global _USER_SETTINGS_ERROR
    _USER_SETTINGS_ERROR = None
    _SAFETY_SETTINGS_WARNINGS[:] = []
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "user_settings.json")
    try:
        if not os.path.exists(path):
            return
        # utf-8-sig: PowerShell 5.1's `Set-Content/Out-File -Encoding utf8`
        # writes a BOM, which plain utf-8 json.load rejects.
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as exc:
        _report_user_settings_failure(path, f"{type(exc).__name__}: {exc}")
        return
    if not isinstance(data, dict):
        _report_user_settings_failure(
            path, f"it holds a {type(data).__name__}, not a JSON object")
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
        if key.startswith("_") or key not in g or key in _ENV_ONLY_KEYS:
            continue          # existing public, non-secret constants only
        cur = g[key]
        if key in _SAFETY_LIST_BASELINE:
            # Inside a try like every other key: this runs at import, so a
            # raise here would stop core.config importing and JARVIS booting.
            # On any error the shipped list stays (2026-10-02 review).
            try:
                g[key] = _merge_safety_list(key, val, cur)
            except Exception:
                pass
            continue
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
