#!/usr/bin/env python3
"""JARVIS Settings — the GUI behind the tray's Settings submenu.

`tray.py` launches this with ``subprocess.Popen([sys.executable,
SETTINGS_WINDOW, "--tab", <name>])`` from each Settings menu item
(Voice/Audio, AI/Models, Privacy/Ambient, Integrations, Advanced) and from the
Audio "Voice / Audio Settings…" item. Until this file existed those six menu
items were silent no-ops — this is the surface they open.

Design notes
────────────
* Dark theme matching the existing tray dialogs (bg ``#0d1117``, fg
  ``#c9d1d9``, Consolas) — see tray.py's `_run_about_dialog`.
* Reads/writes ``data/user_settings.json`` (data/ is gitignored — the schema
  ships as ``tools/user_settings.example.json``). On first run the GUI creates
  ``data/user_settings.json`` from the built-in defaults, which mirror
  ``core/config.py``'s current values.
* The live consumer of these knobs (bobert_companion / core.voice_pipeline._cfg
  etc.) is a SEPARATE, parallel task. This file's only job is to persist a
  valid JSON document; it never imports the monolith.
* Writes use the same atomic temp-file + ``os.replace`` pattern as tray.py's
  ``_send_command`` so a crash mid-save can't corrupt the settings file.
* SECURITY: integration secrets are NEVER read from or written to the repo.
  The Integrations tab shows only PRESENT / not-set status for each key (probed
  from the OS environment) and never displays a secret's value. The plain
  text fields there persist NON-secret connection hints (host/port) to the
  user (gitignored) settings file only.

Everything above the ``# ── GUI ──`` divider is import-safe with no GUI
dependency, so the test-suite can exercise the schema, defaults, load/save
round-trip and CLI parsing on a bare CI runner where tkinter is absent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# ──────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────
# tools/settings_window.py → project root is one level up.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
# Default on-disk location of the live settings document. The actual path used
# by load/save is resolved at CALL time via ``settings_path()`` so a redirect
# (the ``JARVIS_SETTINGS_PATH`` env override below) takes effect even after this
# module is imported — e.g. the test runners point the whole suite at a
# throwaway file so a leaked ``save_settings`` can NEVER clobber the real one.
SETTINGS_PATH = os.path.join(DATA_DIR, "user_settings.json")
# Shipped, tracked template (data/ is fully gitignored).
EXAMPLE_PATH = os.path.join(PROJECT_DIR, "tools", "user_settings.example.json")

# Env var that redirects BOTH load and save away from ``SETTINGS_PATH``. When
# set and non-empty, every read/write (and the atomic temp file derived from it)
# uses this path instead. Unset/blank → the default above, i.e. today's
# behaviour exactly. Tests set this to a temp file; production never sets it.
SETTINGS_PATH_ENV = "JARVIS_SETTINGS_PATH"


def settings_path() -> str:
    """Resolve the settings file path, honouring the ``JARVIS_SETTINGS_PATH``
    override at call time. Returns that env var's value when set and non-empty,
    otherwise the default ``data/user_settings.json``. Resolving here (rather
    than binding a module-level default once at import) is what lets a redirect
    set after import still take effect for both load and save."""
    override = (os.environ.get(SETTINGS_PATH_ENV) or "").strip()
    return override or SETTINGS_PATH

# Theme — identical palette to the tray dialogs.
BG = "#0d1117"
FG = "#c9d1d9"
FIELD_BG = "#161b22"
ACCENT = "#1f6feb"
MUTED = "#8b949e"
FONT = ("Consolas", 10)
FONT_BOLD = ("Consolas", 11, "bold")
FONT_SMALL = ("Consolas", 9)

RESTART_NOTE = "Some changes apply on the next restart."

# Local Ollama endpoint + a STATIC fallback list of common chat tags, used to
# seed the "Local LLM model" dropdown when Ollama is unreachable at GUI-open
# time (so the field still offers sensible choices offline). The live list is
# probed by `installed_ollama_models()`.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_FALLBACK = [
    "gemma4:26b-a4b-it-qat",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
    "qwen2.5:14b-instruct-q5_K_M",
    "llama3.1:8b-instruct-q5_K_M",
]
# Tag substrings that are NOT chat models (embedding / vision) — excluded from
# the chat-model dropdown. Mirrors skills/model_picker's markers.
_NON_CHAT_MARKERS = (
    "nomic-embed", "embed-text", "-embed", "bge-", "all-minilm",
    "vl:", "-vl", "vision", "llava", "moondream", "bakllava",
)


def installed_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Installed Ollama CHAT tags via GET /api/tags (embedding + vision models
    excluded), or the static OLLAMA_MODEL_FALLBACK list if Ollama is
    unreachable. Import-safe: ``requests`` is imported lazily so importing this
    module (for the tests / schema) never requires it or a network call."""
    try:
        import requests  # lazy: keep module import dependency-free
        r = requests.get(f"{base_url}/api/tags", timeout=(2, 3))
        if not r.ok:
            return list(OLLAMA_MODEL_FALLBACK)
        names = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
        chat = [n for n in names
                if not any(mk in n.lower() for mk in _NON_CHAT_MARKERS)]
        return chat or list(OLLAMA_MODEL_FALLBACK)
    except Exception:
        return list(OLLAMA_MODEL_FALLBACK)


# Synthetic mic-picker choices that don't map to a real device index. They are
# part of the MICROPHONE_INDEX contract the monolith already honours (None =
# auto / PREFERRED_INPUT_DEVICES lookup; a NEGATIVE index = hard-off, no capture
# stream is opened — see bobert_companion._mic_input_disabled). The label text
# is what the user sees; the value is what persists to user_settings.json.
MIC_AUTO_LABEL = "System default (auto)"
MIC_OFF_LABEL = "Off (no mic)"
MIC_AUTO_INDEX = None
MIC_OFF_INDEX = -1


def list_input_devices() -> list[tuple[str, int]]:
    """Available audio INPUT devices as ``(label, index)`` tuples, e.g.
    ``("[2] Microphone (Realtek)", 2)``.

    Import-safe and never raises: ``sounddevice`` is imported lazily (it pulls
    in PortAudio), so importing this module for the tests / schema never needs
    it. Mirrors ``installed_ollama_models()``'s lazy-and-tolerant pattern — any
    failure (no sounddevice, no PortAudio, headless CI) returns ``[]`` so the
    picker still renders with only the synthetic auto/off choices.

    Only the device NAME + its query index are read; no stream is opened.
    """
    out: list[tuple[str, int]] = []
    try:
        import sounddevice as sd  # lazy: PortAudio dependency, GUI-only
        for idx, dev in enumerate(sd.query_devices()):
            try:
                if int(dev.get("max_input_channels", 0)) > 0:
                    name = str(dev.get("name", "")).strip() or f"device {idx}"
                    out.append((f"[{idx}] {name}", idx))
            except (TypeError, ValueError, KeyError):
                continue
    except Exception:
        return []
    return out


def mic_choices(saved_index=MIC_AUTO_INDEX) -> list[tuple[str, int]]:
    """The full ordered ``(label, index)`` list for the mic picker combobox:
    the two synthetic choices (auto / off) followed by every live input device.

    If ``saved_index`` is a real device index that is NOT currently present
    (e.g. a USB mic that's unplugged right now), a synthetic
    ``"[idx] (saved device, not connected)"`` row is appended so the saved
    selection stays visible and round-trips — mirroring how the ``combo`` branch
    keeps an unknown saved Ollama tag visible rather than silently dropping it.
    """
    choices: list[tuple[str, int]] = [
        (MIC_AUTO_LABEL, MIC_AUTO_INDEX),
        (MIC_OFF_LABEL, MIC_OFF_INDEX),
    ]
    live = list_input_devices()
    choices.extend(live)
    if isinstance(saved_index, int) and saved_index >= 0 \
            and saved_index not in [i for _, i in live]:
        choices.append((f"[{saved_index}] (saved device, not connected)",
                        saved_index))
    return choices


def mic_index_to_label(index, choices: list[tuple[str, int]]) -> str:
    """Resolve a stored MICROPHONE_INDEX to the combobox label to preselect."""
    for label, idx in choices:
        if idx == index:
            return label
    return MIC_AUTO_LABEL


def mic_label_to_index(label: str, choices: list[tuple[str, int]]):
    """Translate a chosen combobox label back to the int (or None) to persist."""
    for lbl, idx in choices:
        if lbl == label:
            return idx
    return MIC_AUTO_INDEX

# Order matters — drives both the Notebook tab order and `--tab` resolution.
TAB_ORDER = ["voice", "ai", "privacy", "integrations", "advanced"]
TAB_LABELS = {
    "voice": "Voice / Audio",
    "ai": "AI / Models",
    "privacy": "Privacy / Ambient",
    "integrations": "Integrations",
    "advanced": "Advanced",
}

# ──────────────────────────────────────────────────────────────────────────
#  Settings schema
# ──────────────────────────────────────────────────────────────────────────
# A flat dict of JSON-key → field-spec. Each spec is a dict with:
#   tab      one of TAB_ORDER
#   label    human label for the control
#   type     "bool" | "enum" | "str" | "int" | "float" | "text"
#   default  default value (matches core/config.py's current default)
#   help     (optional) one-line hint shown under the control
#   choices  (enum only) list of allowed string values
#   secret_env (integration status rows only) the OS env var probed for
#              PRESENT/not-set; its VALUE is never read into a control
#
# Keeping the schema as plain data (no tkinter) lets the tests assert on
# defaults / coverage and lets `default_settings()` build the template file
# without ever importing the GUI.
#
# This is a SAFE, curated subset of core/config.py — destructive or
# hardware-pinning knobs (CAMERAS, MONITORS, CONFIRM_KEYWORDS, robot IPs) are
# intentionally omitted.
SCHEMA: dict[str, dict] = {
    # ── Voice / Audio ──────────────────────────────────────────────────
    "VOICE_MODE": {
        "tab": "voice", "label": "Voice pipeline", "type": "enum",
        "choices": ["turn_based", "realtime"], "default": "turn_based",
        "help": "realtime = low-latency streaming (needs optional deps; "
                "falls back to turn_based).",
    },
    "WAKE_WORD_AUTOSTART": {
        "tab": "voice", "label": "Neural wake-word in standby", "type": "bool",
        "default": False,
        "help": "Use the neural detector to spot 'Hey JARVIS' while sleeping.",
    },
    "START_IN_STANDBY": {
        "tab": "voice", "label": "Start in wake-word mode (Alexa-style)",
        "type": "bool", "default": False,
        "help": "Boot SILENT; say 'JARVIS' to wake, it answers, then back to "
                "standby — instead of always-listening. Pairs with the neural "
                "wake-word option above.",
    },
    "AMBIENT_MUSIC_REFUSE_WAKE": {
        "tab": "voice", "label": "Require 'JARVIS' while music is playing",
        "type": "bool", "default": True,
        "help": "While music plays, only obey commands that start with 'JARVIS' "
                "(stops replies to song lyrics). Turn OFF if it keeps cutting "
                "you off while your own music is on.",
    },
    "MICROPHONE_INDEX": {
        "tab": "voice", "label": "Microphone", "type": "device",
        "default": None,
        "help": "Which mic to use. 'System default (auto)' follows the "
                "preferred-device list below and auto-switches as devices come "
                "and go; pick a specific device to pin it; 'Off (no mic)' "
                "disables capture entirely. The list is your live input devices "
                "(probed when this window opens).",
    },
    "PREFERRED_INPUT_DEVICES": {
        "tab": "voice", "label": "Preferred mic names (auto mode)",
        "type": "text", "default": [],
        "help": "One device-name substring per line, most-preferred first. In "
                "'System default (auto)' mode JARVIS picks the first connected "
                "match and auto-switches when you plug/unplug. Ignored when a "
                "specific mic is pinned above. Empty = use the OS default.",
    },
    "TTS_VOICE": {
        "tab": "voice", "label": "TTS voice", "type": "str",
        "default": "en-GB-RyanNeural",
        "help": "Edge neural voice name, e.g. en-GB-RyanNeural.",
    },
    "TTS_BACKEND": {
        "tab": "voice", "label": "TTS backend", "type": "enum",
        "choices": ["edge", "pyttsx3", "xtts"], "default": "edge",
        "help": "edge = online neural; pyttsx3 = offline SAPI; xtts = clone.",
    },
    "VOICE_CLONE_ENABLED": {
        "tab": "voice", "label": "Local voice clone (Chatterbox)", "type": "bool",
        "default": False,
        "help": "Speak replies in a CLONED voice (Chatterbox on the 3090). "
                "Needs 'chatterbox-tts' installed + CUDA + a consented profile "
                "selected below. Falls back to the normal voice if anything's "
                "missing — never silences JARVIS. Off by default.",
    },
    "VOICE_CLONE_PROFILE": {
        "tab": "voice", "label": "Active voice-clone profile", "type": "str",
        "default": "",
        "help": "Name of a profile under data/voice_profiles/ (enrolled via "
                "tools/enroll_voice.py with consent). Empty = none selected. "
                "Only a profile with consent=true is used.",
    },
    "VOICE_CLONE_MODEL": {
        "tab": "voice", "label": "Voice-clone engine", "type": "enum",
        "choices": ["chatterbox"], "default": "chatterbox",
        "help": "Local voice-cloning engine. Currently only 'chatterbox'.",
    },
    "AUDIO_PROCESSING_ENABLED": {
        "tab": "voice", "label": "Audio processing (master)", "type": "bool",
        "default": True,
        "help": "Master switch for the mic-cleanup chain below.",
    },
    "AUDIO_ECHO_CANCEL": {
        "tab": "voice", "label": "Echo cancellation (AEC)", "type": "bool",
        "default": True,
        "help": "Cancel JARVIS's own playback from the mic.",
    },
    "AUDIO_NOISE_SUPPRESS": {
        "tab": "voice", "label": "Noise suppression (NS)", "type": "bool",
        "default": True, "help": "Suppress stationary background noise.",
    },
    "AUDIO_AGC": {
        "tab": "voice", "label": "Auto gain control (AGC)", "type": "bool",
        "default": True, "help": "Normalise mic level before STT.",
    },
    "VAD_THRESHOLD": {
        "tab": "voice", "label": "VAD threshold", "type": "float",
        "default": 0.008,
        "help": "Mic RMS to treat as speech; raise to ignore more noise.",
    },
    "AUDIO_DUCKING_ENABLED": {
        "tab": "voice", "label": "Duck other apps while speaking", "type": "bool",
        "default": True,
        "help": "Lower other apps' volume while JARVIS talks (Windows).",
    },
    # 2026-07-08: surface the Whisper STT device/model so the v2.0.23 crash-
    # workaround is settable AND persisted — previously they lived only in
    # core/config.py, so a Settings save (which rewrites user_settings.json from
    # the schema) dropped any hand-set WHISPER_DEVICE. Default stays 'auto'
    # (v2.0.23 made auto crash-safe via the VRAM plan); this is persistence
    # plumbing only and does not touch the runtime whisper code.
    "WHISPER_DEVICE": {
        "tab": "voice", "label": "Whisper STT device", "type": "enum",
        "choices": ["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
        "default": "auto",
        "help": "Where speech-to-text runs. auto = let ctranslate2/torch decide "
                "(crash-safe VRAM plan); cuda / cuda:N pin a GPU (e.g. cuda:1 to "
                "keep the primary card free); cpu forces the legacy path.",
    },
    "WHISPER_MODEL_CUDA": {
        "tab": "voice", "label": "Whisper GPU model", "type": "str",
        "default": "large-v3-turbo",
        "help": "faster-whisper model used on the GPU (~3.1 GB VRAM). "
                "large-v3-turbo is ~8x faster than large-v3 at near-identical "
                "accuracy.",
    },

    # ── AI / Models ────────────────────────────────────────────────────
    "AI_BACKEND": {
        "tab": "ai", "label": "Primary AI backend", "type": "enum",
        "choices": ["claude", "ollama"], "default": "claude",
        "help": "claude = prefer cloud (paid — see per-conversation cost on the "
                "model below); ollama = local-only baseline, $0 per conversation.",
    },
    "CLAUDE_MODEL": {
        "tab": "ai", "label": "Claude model", "type": "enum",
        "choices": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-sonnet-5",
                    "claude-opus-4-6", "claude-opus-4-8"],
        "default": "claude-sonnet-5",
        "help": "Cloud model + est. cost PER CONVERSATION: Haiku ~$0.02, "
                "Sonnet 5 ~$0.06 (default — near-Opus smarts at Sonnet price), "
                "Opus 4.8 ~$0.10 (the ceiling). (Local Ollama is $0 — set the "
                "backend above to ollama.)",
    },
    "LOCAL_LLM_MODEL": {
        "tab": "ai", "label": "Local LLM model (Ollama, $0)", "type": "combo",
        "default": "gemma4:26b-a4b-it-qat",
        "choices": OLLAMA_MODEL_FALLBACK,
        "help": "Ollama tag for the always-on local brain — $0 per conversation. "
                "The list is your installed Ollama chat models (probed when this "
                "window opens); you can also type any tag. Switch by voice too: "
                "'switch to the 32B'.",
    },
    "CLAUDE_OPTIONAL": {
        "tab": "ai", "label": "Claude is optional (never required)",
        "type": "bool", "default": True,
        "help": "A missing/capped Claude key is not treated as a failure.",
    },
    "LOCAL_LLM_FALLBACK": {
        "tab": "ai", "label": "Fall back to local LLM", "type": "bool",
        "default": True,
        "help": "Serve turns on the local model when Claude is unavailable.",
    },
    "LOCAL_VISION_FALLBACK": {
        # 2026-07-08: default MUST mirror core/config.py (False). Was True here,
        # so fresh installs silently enabled the on-demand VLM the config
        # default deliberately leaves off. Kept in lockstep by the AST test in
        # test_settings_window.py that compares every SCHEMA default to the
        # core.config literal.
        "tab": "ai", "label": "Local vision fallback", "type": "bool",
        "default": False,
        "help": "Retry vision on the local VLM when the cloud call fails. "
                "Loads the ~7.3 GB VLM on-demand — see the VRAM budget above.",
    },
    "RAG_ENABLED": {
        "tab": "ai", "label": "Personal RAG (document memory)", "type": "bool",
        "default": True,
        "help": "Index your documents for recall (local nomic-embed-text, "
                "~0.3 GB VRAM). Counts toward the VRAM budget above.",
    },
    "LTM_ENABLED": {
        "tab": "ai", "label": "Long-term memory", "type": "bool",
        "default": True,
        "help": "Record every conversation turn and recall relevant facts "
                "each turn (local embedder, ~0.2 GB VRAM). Off = JARVIS "
                "remembers nothing new between sessions.",
    },
    "STREAMING_TTS_ENABLED": {
        "tab": "voice", "label": "Speak while replies stream", "type": "bool",
        "default": True,
        "help": "Start speaking the first sentence of a cloud reply while "
                "the rest is still generating (faster feel; action commands "
                "are never voiced early).",
    },
    "STREAMING_AUTO_FULLSCREEN": {
        "tab": "ai", "label": "Auto-fullscreen TV shows & movies", "type": "bool",
        "default": True,
        "help": "After a show or movie actually starts playing, send the "
                "player fullscreen ('f' on YouTube / Netflix / Disney+ / "
                "Prime / Hulu / Max). Off = playback starts windowed.",
    },
    "BARGE_IN_ENABLED": {
        "tab": "voice", "label": "Barge-in (interrupt him by voice)",
        "type": "bool", "default": True,
        "help": "Say the wake word while JARVIS is talking to cut him off "
                "and be heard. His own voice can never trigger it.",
    },
    "FOCUS_MODE_ENABLED": {
        "tab": "voice", "label": "Focus mode / do-not-disturb available",
        "type": "bool", "default": True,
        "help": "Let you say 'focus mode' / 'do not disturb' to hold "
                "unsolicited announcements (print, weather, Teams, timers) "
                "until you resume, then hear a recap of what you missed. "
                "Wake-word and command replies always work. This only makes "
                "the feature available — focus mode always starts OFF.",
    },
    "AIR_CONTROL_ENABLED": {
        "tab": "ai", "label": "Air control auto-start (Kinect hand-mouse)",
        "type": "bool", "default": False,
        "help": "Start the Kinect hand-mouse automatically at boot. Off "
                "(default) still allows 'air control on' by voice — this "
                "only controls unattended auto-start.",
    },
    "KINECT_ENABLED": {
        "tab": "ai", "label": "Kinect sensor (presence / gestures)",
        "type": "bool", "default": False,
        "help": "Use the Kinect v2 for presence, gestures and pointing. "
                "Runs on CPU/USB — no extra VRAM.",
    },
    "AIR_MOUSE_REQUIRE_OPEN_PALM": {
        "tab": "ai", "label": "Air-mouse: require an open palm to engage",
        "type": "bool", "default": True,
        "help": "Passive mode only takes the cursor on an OPEN palm raised + held "
                "briefly (fewer false triggers from a closed/pointing hand "
                "reaching or gesturing). Off = height alone can engage.",
    },
    "AIR_MOUSE_ARM_RELAXES_GATE": {
        "tab": "ai", "label": "Air-mouse: 'take the cursor' relaxes the gate",
        "type": "bool", "default": True,
        "help": "When you say 'take the cursor' / 'mouse control on' the strict "
                "smart-pose gate relaxes to height-only so a raised hand engages "
                "right away. Off = still needs the full open-palm hold even armed.",
    },
    "AIR_MOUSE_FIST_RELEASES": {
        "tab": "ai", "label": "Air-mouse: a held fist releases the cursor",
        "type": "bool", "default": False,
        "help": "When ON, holding a closed fist for ~0.6 s lets go of the cursor "
                "without lowering your hand. OFF by default because it fights the "
                "click/drag gesture (a normal close stopped tracking). With it off, "
                "close to click/drag and lower your hand to let go.",
    },
    "AIR_MOUSE_PER_APP_DISABLE": {
        "tab": "ai", "label": "Air-mouse: stand down over fullscreen games/video",
        "type": "bool", "default": True,
        "help": "Automatically disable the air-mouse when a fullscreen game or "
                "video player is in the foreground (edit the app list in "
                "data/user_settings.json → AIR_MOUSE_DISABLED_APP_HINTS).",
    },
    "ENABLE_ORCHESTRATOR": {
        "tab": "ai", "label": "Sub-agent orchestrator", "type": "bool",
        "default": True,
        "help": "Fan standing briefings out to parallel sub-agents.",
    },
    "AMBIENT_LEARNING_FORCE_LOCAL": {
        "tab": "ai", "label": "Free ambient learning (local model)",
        "type": "bool", "default": False,
        "help": "Force ambient / background learning onto the local model so it "
                "costs $0. Foreground conversation is unaffected.",
    },

    # ── Privacy / Ambient ──────────────────────────────────────────────
    "AMBIENT_LISTEN_ENABLED": {
        "tab": "privacy", "label": "Ambient listening", "type": "bool",
        "default": False,
        "help": "Passively transcribe surroundings to learn context. OFF "
                "by default.",
    },
    "AMBIENT_SCREEN_ENABLED": {
        "tab": "privacy", "label": "Ambient screen capture", "type": "bool",
        "default": False,
        "help": "Periodically read the screen for ambient context.",
    },
    "STANDBY_LOOP_ENABLED": {
        "tab": "privacy", "label": "Standby music auto-detect", "type": "bool",
        "default": True,
        "help": "Auto-enter wake-word-only mode when music with lyrics plays.",
    },
    "SCREENSHOT_PRIVACY_BLOCKLIST": {
        "tab": "privacy", "label": "Screenshot privacy blocklist", "type": "text",
        "default": [],
        "help": "One app/window title substring per line; screen vision skips "
                "any window whose title matches (case-insensitive). Empty = "
                "off. Try: 1password, bitwarden, keepass, banking.",
    },
    "DAILY_BUDGET_USD": {
        "tab": "privacy", "label": "Daily Claude $ cap", "type": "float",
        "default": 1.0,
        "help": "Soft daily ceiling for the Chappie continuous-learning loop's "
                "Claude API spend (USD).",
    },
    "DEEP_AUDIT_BUDGET_USD": {
        "tab": "privacy", "label": "Deep-audit daily $ cap", "type": "float",
        "default": 5.0,
        "help": "Daily ceiling for the background deep-audit diagnostic (USD). "
                "The JARVIS_DEEP_AUDIT_BUDGET_USD env var overrides this.",
    },

    # ── Integrations ───────────────────────────────────────────────────
    # Status-only rows (probe env presence, never show the value):
    "_status_anthropic": {
        "tab": "integrations", "label": "Anthropic Claude API", "type": "status",
        "secret_env": ["ANTHROPIC_API_KEY"],
    },
    "_status_porcupine": {
        "tab": "integrations", "label": "Porcupine wake word", "type": "status",
        "secret_env": ["PORCUPINE_ACCESS_KEY"],
    },
    "_status_azure_tts": {
        "tab": "integrations", "label": "Azure TTS", "type": "status",
        "secret_env": ["AZURE_TTS_KEY", "AZURE_TTS_REGION"],
    },
    "_status_elevenlabs": {
        "tab": "integrations", "label": "ElevenLabs TTS", "type": "status",
        "secret_env": ["ELEVENLABS_API_KEY"],
    },
    "_status_bambu": {
        "tab": "integrations", "label": "Bambu Lab printer", "type": "status",
        "secret_env": ["BAMBU_PRINTER_IP", "BAMBU_ACCESS_CODE", "BAMBU_SERIAL"],
    },
    "_status_govee": {
        "tab": "integrations", "label": "Govee smart home", "type": "status",
        "secret_env": ["GOVEE_API_KEY"],
    },
    "_status_hue": {
        "tab": "integrations", "label": "Philips Hue", "type": "status",
        "secret_env": [],
        "config_file": "sh_hue_config.json",
        "help": "Configured via data/sh_hue_config.json (bridge IP + button).",
    },
    "_status_obs": {
        "tab": "integrations", "label": "OBS Studio", "type": "status",
        "secret_env": ["OBS_HOST", "OBS_PASSWORD"],
    },
    "_status_deco": {
        "tab": "integrations", "label": "TP-Link Deco router", "type": "status",
        "secret_env": ["DECO_HOST", "DECO_PASSWORD"],
    },
    "_status_phone": {
        "tab": "integrations", "label": "Phone bridge / push", "type": "status",
        "secret_env": ["TELEGRAM_BOT_TOKEN", "NTFY_TOPIC", "PUSHOVER_TOKEN"],
        "help": "PRESENT if any one channel (Telegram / ntfy / Pushover) is set.",
        "match": "any",
    },
    # Non-secret connection hints persisted to user_settings.json:
    "OBS_HOST_HINT": {
        "tab": "integrations", "label": "OBS host (non-secret)", "type": "str",
        "default": "",
        "help": "e.g. localhost. Stored in user_settings.json, not the repo. "
                "Leave the password in your .env / OS environment.",
    },
    "OBS_PORT_HINT": {
        "tab": "integrations", "label": "OBS port (non-secret)", "type": "str",
        "default": "",
        "help": "e.g. 4455. Non-secret — safe to keep here.",
    },
    "HUE_BRIDGE_IP_HINT": {
        "tab": "integrations", "label": "Hue bridge IP (non-secret)",
        "type": "str", "default": "",
        "help": "Local bridge IP, e.g. 192.168.1.50. Non-secret hint only.",
    },

    # ── Advanced ───────────────────────────────────────────────────────
    "HUD_ENABLED": {
        "tab": "advanced", "label": "On-screen HUD", "type": "bool",
        "default": True, "help": "Drives the unified HUD at boot.",
    },
    "TRAY_ENABLED": {
        "tab": "advanced", "label": "System-tray applet", "type": "bool",
        "default": True,
    },
    "RETICLE_OVERLAY_ENABLED": {
        "tab": "advanced", "label": "Click-feedback reticle", "type": "bool",
        "default": True,
        "help": "Brief target flash where a UI-automation action fires.",
    },
    "MODEL_ROUTING": {
        "tab": "ai", "label": "Per-function model", "type": "routing",
        "default": {"chat": "auto", "vision": "auto", "ambient": "auto"},
        "choices": ["auto", "local", "cloud"],
        "help": "Pick the brain per function: local (free Ollama) / cloud "
                "(Claude) / auto (cloud, local on failure). vision = see screen, "
                "chat = conversation, ambient = background learning.",
    },
    "PUSHBACK_ENABLED": {
        "tab": "advanced", "label": "JARVIS-style pushback", "type": "bool",
        "default": True,
        "help": "In-character objection before gray-zone bulk actions.",
    },
    "MISSION_NARRATION_ENABLED": {
        "tab": "advanced", "label": "Mission narration", "type": "bool",
        "default": True,
        "help": "Narrate multi-step action chains aloud.",
    },
    "OVERNIGHT_UPGRADE_ENABLED": {
        "tab": "advanced", "label": "Overnight self-upgrade", "type": "bool",
        "default": False,
        "help": "Auto-fire the upgrade pipeline when idle (currently paused).",
    },
    "VAD_DEBUG": {
        "tab": "advanced", "label": "VAD debug prints", "type": "bool",
        "default": True,
        "help": "Print peak RMS after each utterance to tune VAD_THRESHOLD.",
    },
    "SCREEN_VISION_ENABLED": {
        "tab": "advanced", "label": "Screen vision", "type": "bool",
        "default": True,
        "help": "Let JARVIS see and reason about the screen.",
    },
    "PC_CONTROL_ENABLED": {
        "tab": "advanced", "label": "PC control", "type": "bool",
        "default": True,
        "help": "Allow launching apps, opening URLs, etc.",
    },
    # ── Live web interface (tools/web_interface.py) ─────────────────────
    # A local-LAN dashboard to watch JARVIS and type commands to him. The
    # typed command runs through the SAME inject channel as a spoken one, so
    # anyone who can reach the bound socket can DRIVE JARVIS — hence the token
    # requirement on a non-local bind, spelled out in the help text below.
    "WEB_INTERFACE_ENABLED": {
        "tab": "advanced", "label": "Live web interface", "type": "bool",
        "default": False,
        "help": "Serve a local web dashboard at boot to watch JARVIS and type "
                "commands. Off (default) still allows 'start the web interface' "
                "by voice. SECURITY: a typed command runs like a spoken one, so "
                "keep the bind on localhost unless you set a token below.",
    },
    "WEB_INTERFACE_PORT": {
        "tab": "advanced", "label": "Web interface port", "type": "int",
        "default": 8766,
        "help": "TCP port for the dashboard (do not use 8443 — that's the "
                "AirTag tracker).",
    },
    "WEB_INTERFACE_BIND": {
        "tab": "advanced", "label": "Web interface bind address", "type": "str",
        "default": "127.0.0.1",
        "help": "127.0.0.1 = localhost only (safe, nothing off-box can reach "
                "it). 0.0.0.0 or a LAN IP EXPOSES it to your whole network and "
                "REQUIRES a token below — the server refuses to start otherwise.",
    },
    "WEB_INTERFACE_TOKEN": {
        "tab": "advanced", "label": "Web interface token", "type": "str",
        "default": "",
        "help": "Shared secret required on every request when set (and MANDATORY "
                "for a non-localhost bind). Treat it like a password — anyone "
                "with it can command JARVIS from any device on your LAN.",
    },
}

# Field types whose key is a real persisted setting (everything except the
# read-only "status" rows, whose keys start with "_status_").
_PERSISTED_TYPES = {"bool", "enum", "str", "combo", "int", "float", "text",
                    "routing", "device"}


def persisted_keys() -> list[str]:
    """The schema keys that map to a stored value (excludes status rows)."""
    return [k for k, s in SCHEMA.items() if s.get("type") in _PERSISTED_TYPES]


def default_settings() -> dict:
    """The full template: every persisted key at its schema default.

    Mirrors core/config.py's current defaults so a fresh install boots with a
    valid file. Mutable defaults (lists) are deep-copied so callers can't
    mutate the schema by editing the result.
    """
    out: dict = {}
    for key in persisted_keys():
        default = SCHEMA[key]["default"]
        if isinstance(default, list):
            default = list(default)
        elif isinstance(default, dict):
            default = dict(default)
        out[key] = default
    return out


def coerce_value(spec: dict, raw):
    """Coerce a raw value to the type its schema spec declares.

    Never raises on bad input — falls back to the spec default so a hand-edited
    settings file with a typo can't crash the GUI or a downstream reader.
    """
    typ = spec.get("type")
    default = spec.get("default")
    try:
        if typ == "bool":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(raw)
            if isinstance(raw, str):
                return raw.strip().lower() in ("1", "true", "yes", "on", "y")
            return bool(default)
        if typ == "int":
            return int(raw)
        if typ == "float":
            return float(raw)
        if typ == "enum":
            val = str(raw)
            choices = spec.get("choices") or []
            return val if val in choices else default
        if typ == "combo":
            # Free-form string with SUGGESTED choices: unlike enum, a value
            # outside `choices` is allowed (the user may type any Ollama tag).
            return str(raw)
        if typ == "device":
            # A device index: int | None, matching the MICROPHONE_INDEX
            # contract (None = auto / preferred-list lookup, a negative index =
            # hard-off). Persist the INT (or None), never the friendly label —
            # so what lands in user_settings.json is exactly what
            # core.config._apply_user_settings coerces back. null / "" / None ->
            # the default (None); 3 / "3" / -1 -> int; a non-numeric value falls
            # through to the default rather than raising.
            if raw is None:
                return default
            if isinstance(raw, bool):           # guard: bool is an int subclass
                return default
            if isinstance(raw, str) and raw.strip() == "":
                return default
            return int(raw)
        if typ == "text":
            # Stored as a list of lines; accept a list or a newline string.
            if isinstance(raw, list):
                return [str(x) for x in raw]
            if isinstance(raw, str):
                return [ln.strip() for ln in raw.splitlines() if ln.strip()]
            return list(default) if isinstance(default, list) else []
        if typ == "routing":
            # nested {function: route}; merge valid entries over the default,
            # drop unknown functions / invalid routes.
            base = dict(default) if isinstance(default, dict) else {}
            opts = spec.get("choices") or ["auto", "local", "cloud"]
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if k in base and str(v) in opts:
                        base[k] = str(v)
            return base
        # str (and anything unknown) → string
        return str(raw)
    except (TypeError, ValueError):
        if isinstance(default, list):
            return list(default)
        if isinstance(default, dict):
            return dict(default)
        return default


def load_settings(path: str | None = None) -> dict:
    """Load settings, layering the on-disk file over the defaults.

    Missing file or missing keys fall back to defaults; every value is coerced
    to its schema type. Unknown keys in the file are preserved untouched (so a
    newer JARVIS that wrote extra keys isn't clobbered by an older GUI).

    ``path`` defaults to ``settings_path()`` (honours ``JARVIS_SETTINGS_PATH``).
    """
    if path is None:
        path = settings_path()
    merged = default_settings()
    raw: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                decoded = json.loads(text)
                if isinstance(decoded, dict):
                    raw = decoded
        except (OSError, ValueError):
            raw = {}
    # Coerce known keys; pass through unknown keys verbatim.
    for key, value in raw.items():
        spec = SCHEMA.get(key)
        merged[key] = coerce_value(spec, value) if spec else value
    return merged


def atomic_write_json(path: str, data: dict) -> None:
    """Write `data` as pretty JSON via temp-file + os.replace.

    Same crash-safe pattern as tray.py's `_send_command`: write to a temp file
    in the destination directory, fsync-free, then atomically rename over the
    target so a reader never observes a half-written file.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp", prefix="usettings_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def save_settings(values: dict, path: str | None = None) -> None:
    """Persist `values` (coerced to schema types) to `path`, atomically.

    ``path`` defaults to ``settings_path()`` so a ``JARVIS_SETTINGS_PATH``
    redirect sends the write (and its atomic temp file, derived from this path's
    directory) to the throwaway file instead of the real one.
    """
    if path is None:
        path = settings_path()
    out = default_settings()
    for key, value in values.items():
        spec = SCHEMA.get(key)
        out[key] = coerce_value(spec, value) if spec else value
    atomic_write_json(path, out)


def ensure_settings_file(path: str | None = None) -> dict:
    """Guarantee a valid settings file exists, creating it from defaults.

    Returns the loaded settings. Called on GUI launch so a fresh install lands
    a complete data/user_settings.json on first open. ``path`` defaults to
    ``settings_path()`` (honours ``JARVIS_SETTINGS_PATH``).
    """
    if path is None:
        path = settings_path()
    if not os.path.exists(path):
        try:
            atomic_write_json(path, default_settings())
        except OSError:
            pass
    return load_settings(path)


def integration_status(spec: dict) -> tuple[bool, str]:
    """Resolve a status row to (present, detail) WITHOUT exposing any secret.

    Only the PRESENCE of each env var (or config file) is checked — values are
    never read into the return. `match == "any"` means present if ANY listed
    env var is set; otherwise ALL must be set.
    """
    envs = spec.get("secret_env") or []
    present_flags = [bool((os.environ.get(e) or "").strip()) for e in envs]
    cfg_present = False
    cfg_file = spec.get("config_file")
    if cfg_file:
        cfg_present = os.path.exists(os.path.join(DATA_DIR, cfg_file))

    if not envs and cfg_file:
        return (cfg_present, "configured" if cfg_present else "not configured")
    if not envs and not cfg_file:
        return (False, "not configured")

    if spec.get("match") == "any":
        present = any(present_flags) or cfg_present
    else:
        present = all(present_flags) or (cfg_present and not envs)
    return (present, "present" if present else "not set")


# ──────────────────────────────────────────────────────────────────────────
#  VRAM budget bridge  (import-safe: the engine is stdlib-only and never
#  raises; the heavy work lives in core/vram_budget.py)
# ──────────────────────────────────────────────────────────────────────────
# Keys whose LIVE value feeds the VRAM budget — changing any of these re-runs
# the prediction in the GUI. ``MODEL_ROUTING::vision`` is the flattened Tk var
# name the routing combobox uses (see the routing widget + _collect()).
VRAM_WATCH_KEYS = (
    "LOCAL_LLM_MODEL",
    "MODEL_ROUTING::vision",
    "LOCAL_VISION_FALLBACK",
    "SCREEN_VISION_ENABLED",
    "RAG_ENABLED",
    "KINECT_ENABLED",
)


def _load_vram_budget():
    """Import core.vram_budget lazily, returning the module or None.

    Kept out of module import so the test surface / a bare runner never needs
    it, and so a missing/broken engine simply hides the budget panel rather than
    breaking the whole Settings window."""
    try:
        from core import vram_budget  # lazy: GUI-only dependency
        return vram_budget
    except Exception:
        return None


def budget_from_live_values(values: dict, total_mb=None) -> dict | None:
    """Run the VRAM prediction from a flat dict of CURRENT widget values.

    This is the value→budget function the live GUI callback calls (and the one
    the tests exercise — no Tk/pixels involved). ``values`` is the raw widget
    state: ``LOCAL_LLM_MODEL`` (tag string), the bool toggles, and the
    flattened ``MODEL_ROUTING::vision`` route. Bools may arrive as real bools or
    as strings/ints (Tk ``StringVar``), which the engine coerces. Returns the
    predict_budget() dict, or None when the engine is unavailable. Never raises."""
    vb = _load_vram_budget()
    if vb is None:
        return None
    settings = dict(values) if isinstance(values, dict) else {}
    try:
        return vb.predict_budget(settings, total_mb=total_mb)
    except Exception:
        return None


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI args. `--tab <name>` selects the starting tab."""
    parser = argparse.ArgumentParser(
        prog="settings_window",
        description="JARVIS settings GUI (launched from the tray).",
    )
    parser.add_argument(
        "--tab", choices=TAB_ORDER, default=None,
        help="Which tab to open first (default: the first tab).",
    )
    return parser.parse_args(argv)


def resolve_start_tab(tab) -> int:
    """Map a `--tab` name to its index in TAB_ORDER (0 when unset/unknown)."""
    if tab in TAB_ORDER:
        return TAB_ORDER.index(tab)
    return 0


# ──────────────────────────────────────────────────────────────────────────
#  ── GUI ──   (everything below requires tkinter; kept out of import-time)
# ──────────────────────────────────────────────────────────────────────────
def run_gui(start_tab: int = 0) -> int:
    """Build and run the settings window. Returns a process exit code.

    Imports tkinter lazily so importing this module (for the tests, or for the
    schema) never requires a display or the Tk libraries.
    """
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as exc:  # pragma: no cover - headless/no-Tk path
        sys.stderr.write(f"settings_window: tkinter unavailable ({exc})\n")
        return 2

    settings = ensure_settings_file()

    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - no display
        sys.stderr.write(f"settings_window: no display ({exc})\n")
        return 2

    root.title("JARVIS Settings")
    root.configure(bg=BG)
    root.geometry("640x620")
    root.minsize(560, 480)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    # ttk dark theme.
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=FIELD_BG, foreground=FG,
                    padding=(12, 6), font=FONT)
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG, font=FONT)
    style.configure("Help.TLabel", background=BG, foreground=MUTED,
                    font=FONT_SMALL)
    style.configure("Head.TLabel", background=BG, foreground=FG, font=FONT_BOLD)
    style.configure("TCheckbutton", background=BG, foreground=FG, font=FONT)
    style.map("TCheckbutton", background=[("active", BG)])
    style.configure("TCombobox", fieldbackground=FIELD_BG, background=FIELD_BG,
                    foreground=FG, font=FONT)
    style.configure("TButton", background=FIELD_BG, foreground=FG, font=FONT,
                    padding=(10, 5))
    style.map("TButton", background=[("active", ACCENT)])

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=(10, 4))

    # Tk variable per persisted key; widgets read/write these.
    vars_by_key: dict = {}
    text_widgets: dict = {}
    # Mic-picker (type "device") widgets: key -> (StringVar of the chosen label,
    # [(label, index)] choices). _collect() translates the label back to the int
    # index to persist. Kept separate from vars_by_key because that path stores
    # the var's raw string value, but a device must persist its int, not the
    # friendly label.
    device_widgets: dict = {}

    # VRAM budget panel — widgets populated when the AI tab is built; the engine
    # is loaded lazily and may be absent (no nvidia-smi / import error), in which
    # case the panel shows a muted note and never blocks the window.
    vram = _load_vram_budget()
    vram_widgets: dict = {}
    # Total card VRAM, probed ONCE here so the live recompute doesn't shell out
    # to nvidia-smi on every keystroke. Falls back to the 24 GB default inside
    # the engine when the probe fails.
    vram_total_mb = None
    if vram is not None:
        try:
            vram_total_mb = vram.total_vram_mb()
        except Exception:
            vram_total_mb = None

    def _vram_color(pct: float, over: bool) -> str:
        """Bar colour by load: green <80%, amber 80–100%, red >100% (over)."""
        if over or pct > 100.0:
            return "#f85149"   # red
        if pct >= 80.0:
            return "#d29922"   # amber
        return "#3fb950"       # green

    def _live_vram_values() -> dict:
        """Snapshot the CURRENT widget values the budget depends on, as a flat
        dict for budget_from_live_values(). Reads the live Tk vars so the bar
        reflects unsaved edits. Falls back to the loaded ``settings`` value for
        any var not yet built (e.g. its tab isn't constructed)."""
        out: dict = {}
        for key in VRAM_WATCH_KEYS:
            var = vars_by_key.get(key)
            if var is not None:
                try:
                    out[key] = var.get()
                    continue
                except Exception:
                    pass
            # Fall back to the on-disk/default value (handle the routing subkey).
            if "::" in key:
                base, fn = key.split("::", 1)
                cur = settings.get(base)
                if isinstance(cur, dict):
                    out[key] = cur.get(fn)
            elif key in settings:
                out[key] = settings.get(key)
        return out

    def update_budget(*_a) -> None:
        """Recompute the prediction from live widget values and repaint the bar,
        numeric readout, per-component breakdown and over-budget warning. Bound
        to every VRAM_WATCH_KEYS var so any change updates it instantly. Never
        raises — a failure just leaves the last drawn state."""
        if not vram_widgets:
            return
        try:
            b = budget_from_live_values(_live_vram_values(), total_mb=vram_total_mb)
            if b is None:
                return
            pct = b["pct"]
            color = _vram_color(pct, b["over"])
            # Bar: redraw the filled rectangle to the clamped fraction.
            canvas = vram_widgets.get("canvas")
            if canvas is not None:
                cw = max(1, int(canvas.winfo_width() or 0))
                if cw <= 1:
                    cw = 560  # not yet laid out — use the nominal width
                frac = 0.0
                if b["budget_mb"] > 0:
                    frac = min(1.0, b["total_mb"] / b["budget_mb"])
                canvas.coords(vram_widgets["bar_fill"], 0, 0, int(cw * frac), 18)
                canvas.itemconfigure(vram_widgets["bar_fill"], fill=color)
            # Numeric "20.6 / 24 GB peak (137%)".
            used = b["total_mb"] / 1024.0
            cap = b["total_card_mb"] / 1024.0
            vram_widgets["num"].configure(
                text=f"{used:.1f} / {cap:.0f} GB peak  ({pct:.0f}%)",
                fg=color)
            # Per-component breakdown: "32B 22 · vision 7.3 (on-demand) · …".
            parts = []
            for c in b["components"]:
                gb = c["mb"] / 1024.0
                gb_s = f"{int(round(gb))}" if abs(gb - round(gb)) < 0.05 else f"{gb:.1f}"
                tag = " (on-demand)" if c.get("ondemand") else ""
                parts.append(f"{c['label']} {gb_s}{tag}")
            vram_widgets["parts"].configure(text=" · ".join(parts))
            # Warning row — shown only when over budget.
            warn = vram_widgets.get("warn")
            if warn is not None:
                if b["over"]:
                    warn.configure(text=vram.over_warning(b))
                    warn.grid()
                else:
                    warn.configure(text="")
                    warn.grid_remove()
        except Exception:
            pass

    def _build_vram_panel(parent, row: int) -> int:
        """Build the 'GPU / VRAM budget' panel into ``parent`` at grid ``row``;
        returns the next free row. When the engine is unavailable, drops a single
        muted note instead so the rest of the tab still renders."""
        head = ttk.Label(parent, text="GPU / VRAM budget", style="Head.TLabel")
        head.grid(row=row, column=0, columnspan=2, sticky="w", padx=2,
                  pady=(2, 2))
        row += 1
        if vram is None:
            ttk.Label(parent,
                      text="(VRAM estimate unavailable — core.vram_budget "
                           "could not load.)",
                      style="Help.TLabel", wraplength=560).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=2, pady=(0, 6))
            return row + 1
        # The bar — a thin Canvas with a background track + a coloured fill rect.
        bar = tk.Canvas(parent, height=18, bg=FIELD_BG, highlightthickness=1,
                        highlightbackground="#30363d", bd=0)
        bar.grid(row=row, column=0, columnspan=2, sticky="we", padx=2, pady=(0, 2))
        fill = bar.create_rectangle(0, 0, 0, 18, fill="#3fb950", width=0)
        vram_widgets["canvas"] = bar
        vram_widgets["bar_fill"] = fill
        # Redraw the fill when the bar is first laid out / resized.
        bar.bind("<Configure>", update_budget)
        row += 1
        # Numeric readout + per-component breakdown.
        num = tk.Label(parent, text="", bg=BG, fg=FG, font=FONT, anchor="w")
        num.grid(row=row, column=0, columnspan=2, sticky="w", padx=2, pady=(0, 0))
        vram_widgets["num"] = num
        row += 1
        parts = tk.Label(parent, text="", bg=BG, fg=MUTED, font=FONT_SMALL,
                         anchor="w", justify="left", wraplength=560)
        parts.grid(row=row, column=0, columnspan=2, sticky="w", padx=2,
                   pady=(0, 2))
        vram_widgets["parts"] = parts
        row += 1
        # Over-budget warning (hidden until `over`).
        warn = tk.Label(parent, text="", bg=BG, fg="#f85149", font=FONT_SMALL,
                        anchor="w", justify="left", wraplength=560)
        warn.grid(row=row, column=0, columnspan=2, sticky="w", padx=2,
                  pady=(0, 6))
        warn.grid_remove()
        vram_widgets["warn"] = warn
        row += 1
        return row

    def _add_help(parent, spec, row):
        help_text = spec.get("help")
        if help_text:
            ttk.Label(parent, text=help_text, style="Help.TLabel",
                      wraplength=560, justify="left").grid(
                row=row, column=0, columnspan=2, sticky="w",
                padx=(2, 0), pady=(0, 8))

    def _build_tab(tab_key):
        # Scrollable frame so long tabs (Advanced) don't clip.
        outer = ttk.Frame(notebook, style="TFrame")
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical",
                                  command=canvas.yview)
        inner = ttk.Frame(canvas, style="TFrame")
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse-wheel scrolling, scoped to the canvas the pointer is over.
        # `bind_all` is application-wide and was re-bound per tab, so the LAST
        # tab built (Advanced) captured the wheel for EVERY tab — on the other
        # tabs the wheel scrolled the hidden Advanced canvas (B17). Binding the
        # global wheel only while the pointer is inside THIS canvas (and
        # releasing it on leave) makes each tab scroll its own content.
        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_wheel)

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")

        for w in (canvas, inner):
            w.bind("<Enter>", _bind_wheel)
            w.bind("<Leave>", _unbind_wheel)

        row = 0
        inner.columnconfigure(1, weight=1)
        # The VRAM budget panel lives at the top of the AI tab, beside the
        # model dropdown — like a game's graphics-settings estimator.
        if tab_key == "ai":
            row = _build_vram_panel(inner, row)
        for key, spec in SCHEMA.items():
            if spec.get("tab") != tab_key:
                continue
            typ = spec.get("type")
            label = spec.get("label", key)

            if typ == "status":
                present, detail = integration_status(spec)
                dot = "●" if present else "○"
                color = "#3fb950" if present else MUTED
                badge = tk.Label(inner, text=f"{dot} {label}: {detail}",
                                 bg=BG, fg=color, font=FONT, anchor="w")
                badge.grid(row=row, column=0, columnspan=2, sticky="w",
                           padx=2, pady=(2, 0))
                row += 1
                _add_help(inner, spec, row)
                row += 1
                continue

            if typ == "bool":
                var = tk.BooleanVar(value=bool(settings.get(key)))
                vars_by_key[key] = var
                ttk.Checkbutton(inner, text=label, variable=var).grid(
                    row=row, column=0, columnspan=2, sticky="w",
                    padx=2, pady=(4, 0))
                row += 1
                _add_help(inner, spec, row)
                row += 1
                continue

            if typ == "text":
                ttk.Label(inner, text=label, style="Head.TLabel").grid(
                    row=row, column=0, columnspan=2, sticky="w",
                    padx=2, pady=(6, 2))
                row += 1
                txt = tk.Text(inner, height=5, width=48, bg=FIELD_BG, fg=FG,
                              insertbackground=FG, font=FONT,
                              relief="flat", padx=6, pady=4)
                cur = settings.get(key) or []
                if isinstance(cur, list):
                    txt.insert("1.0", "\n".join(str(x) for x in cur))
                else:
                    txt.insert("1.0", str(cur))
                txt.grid(row=row, column=0, columnspan=2, sticky="we",
                         padx=2, pady=(0, 2))
                text_widgets[key] = txt
                row += 1
                _add_help(inner, spec, row)
                row += 1
                continue

            if typ == "routing":
                # one dropdown per function (vision/chat/ambient), composed back
                # into the MODEL_ROUTING dict on save via the "::" sub-keys.
                ttk.Label(inner, text=label).grid(
                    row=row, column=0, sticky="w", padx=2, pady=(6, 2))
                row += 1
                cur = settings.get(key)
                cur = cur if isinstance(cur, dict) else {}
                opts = spec.get("choices") or ["auto", "local", "cloud"]
                for fn in spec.get("default", {}):
                    ttk.Label(inner, text=f"    • {fn}").grid(
                        row=row, column=0, sticky="w", padx=12, pady=(0, 2))
                    rvar = tk.StringVar(value=str(cur.get(fn, spec["default"][fn])))
                    vars_by_key[f"{key}::{fn}"] = rvar
                    ttk.Combobox(inner, textvariable=rvar, state="readonly",
                                 values=opts, width=28).grid(
                        row=row, column=1, sticky="we", padx=2, pady=(0, 2))
                    row += 1
                _add_help(inner, spec, row)
                row += 1
                continue

            if typ == "combo":
                # Editable dropdown: live-probed Ollama models as suggestions,
                # but the user can still type any tag (not state="readonly").
                ttk.Label(inner, text=label).grid(
                    row=row, column=0, sticky="w", padx=2, pady=(6, 2))
                values = list(spec.get("choices") or [])
                try:
                    values = installed_ollama_models()
                except Exception:
                    pass
                cur_val = str(settings.get(key, ""))
                if cur_val and cur_val not in values:
                    values = [cur_val] + values  # keep a custom saved tag visible
                var = tk.StringVar(value=cur_val)
                vars_by_key[key] = var
                ttk.Combobox(inner, textvariable=var, values=values,
                             width=28).grid(
                    row=row, column=1, sticky="we", padx=2, pady=(6, 2))
                row += 1
                _add_help(inner, spec, row)
                row += 1
                continue

            if typ == "device":
                # Readonly dropdown of live input devices + the synthetic
                # auto/off choices. The combobox shows friendly LABELS; the int
                # index (or None) is recovered in _collect() via device_widgets.
                ttk.Label(inner, text=label).grid(
                    row=row, column=0, sticky="w", padx=2, pady=(6, 2))
                saved = settings.get(key, spec.get("default"))
                try:
                    saved = int(saved) if saved is not None else None
                except (TypeError, ValueError):
                    saved = None
                choices = mic_choices(saved)
                labels = [lbl for lbl, _ in choices]
                var = tk.StringVar(value=mic_index_to_label(saved, choices))
                device_widgets[key] = (var, choices)
                ttk.Combobox(inner, textvariable=var, state="readonly",
                             values=labels, width=28).grid(
                    row=row, column=1, sticky="we", padx=2, pady=(6, 2))
                row += 1
                _add_help(inner, spec, row)
                row += 1
                continue

            # enum / str / int / float → label + control on one row.
            ttk.Label(inner, text=label).grid(
                row=row, column=0, sticky="w", padx=2, pady=(6, 2))
            if typ == "enum":
                var = tk.StringVar(value=str(settings.get(key, "")))
                vars_by_key[key] = var
                ttk.Combobox(inner, textvariable=var, state="readonly",
                             values=spec.get("choices", []), width=28).grid(
                    row=row, column=1, sticky="we", padx=2, pady=(6, 2))
            else:
                var = tk.StringVar(value=str(settings.get(key, "")))
                vars_by_key[key] = var
                tk.Entry(inner, textvariable=var, bg=FIELD_BG, fg=FG,
                         insertbackground=FG, font=FONT, relief="flat").grid(
                    row=row, column=1, sticky="we", padx=2, pady=(6, 2))
            row += 1
            _add_help(inner, spec, row)
            row += 1

        notebook.add(outer, text=TAB_LABELS[tab_key])

    for tab_key in TAB_ORDER:
        _build_tab(tab_key)

    # Live recompute: every VRAM-budget input re-runs the prediction the moment
    # it changes — exactly like a game's graphics menu updating its estimate as
    # you toggle settings. Bound here (after ALL tabs are built) so vars from
    # other tabs (e.g. SCREEN_VISION_ENABLED on Advanced) are already created.
    if vram_widgets:
        for key in VRAM_WATCH_KEYS:
            var = vars_by_key.get(key)
            if var is None:
                continue
            try:
                var.trace_add("write", update_budget)
            except Exception:
                pass
        # Draw the initial state (deferred so the bar Canvas has its real width).
        try:
            root.after(0, update_budget)
        except Exception:
            update_budget()

    try:
        notebook.select(start_tab)
    except Exception:
        pass

    # ── bottom bar: note + buttons ──
    bar = ttk.Frame(root, style="TFrame")
    bar.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Label(bar, text=RESTART_NOTE, style="Help.TLabel").pack(side="left")

    status_var = tk.StringVar(value="")
    ttk.Label(bar, textvariable=status_var, style="Help.TLabel").pack(
        side="left", padx=10)

    def _collect() -> dict:
        out: dict = dict(settings)  # keep unknown/passthrough keys
        for key, var in vars_by_key.items():
            out[key] = var.get()
        for key, widget in text_widgets.items():
            out[key] = widget.get("1.0", "end")
        # Device pickers persist the int index, not the chosen friendly label.
        for key, (var, choices) in device_widgets.items():
            out[key] = mic_label_to_index(var.get(), choices)
        # Fold routing sub-keys ("MODEL_ROUTING::vision") into their nested dict.
        for compound in [c for c in list(out) if "::" in c]:
            base, fn = compound.split("::", 1)
            val = out.pop(compound)
            if not isinstance(out.get(base), dict):
                out[base] = {}
            out[base][fn] = val
        return out

    def _on_save():
        try:
            save_settings(_collect())
            status_var.set("Saved.")
        except Exception as exc:
            try:
                messagebox.showerror("JARVIS Settings",
                                     f"Could not save settings:\n{exc}")
            except Exception:
                pass
            status_var.set("Save failed.")

    def _open_json():
        try:
            ensure_settings_file()
            target = settings_path()
            if hasattr(os, "startfile"):
                os.startfile(target)  # noqa: S606 (Windows-only)
            else:
                status_var.set(target)
        except Exception as exc:
            status_var.set(f"Open failed: {exc}")

    ttk.Button(bar, text="Close", command=root.destroy).pack(
        side="right", padx=(6, 0))
    ttk.Button(bar, text="Save", command=_on_save).pack(side="right")
    ttk.Button(bar, text="Open user_settings.json",
               command=_open_json).pack(side="right", padx=(0, 6))

    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_gui(resolve_start_tab(args.tab))


if __name__ == "__main__":
    raise SystemExit(main())
