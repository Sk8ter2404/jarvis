#!/usr/bin/env python3
"""JARVIS Settings — the GUI behind the tray's Settings submenu.

``tray.py`` launches this with ``subprocess.Popen([sys.executable,
SETTINGS_WINDOW, "--tab", <name>])``. Run as a SCRIPT, Python puts ``tools\\``
(this file's folder) first on ``sys.path``, not the project root — so the
first thing below puts the project root back, or every lazy ``from core
import …`` (the VRAM panel, the chat↔vision lockstep) quietly fails.
``--selftest`` proves the imports resolve without opening a window.

Design notes
────────────
* Dark theme matching the tray dialogs (bg ``#0d1117``, fg ``#c9d1d9``,
  Consolas), including the read-only dropdowns and their lists.
* Reads/writes ``data/user_settings.json`` (gitignored; the shipped template is
  ``tools/user_settings.example.json``). ``core/config.py`` applies that file
  ONCE, when JARVIS starts — so almost every change here takes effect on the
  next start. The window says so, and offers "Save & restart JARVIS" (through
  the same tray command inbox the tray's Restart item uses).
* Save writes ONLY the fields you changed, merged into the file as it is on
  disk at that moment. A voice command that changed a setting while the window
  was open keeps its value, untouched defaults are never frozen into the file,
  and keys this window does not show (CAMERAS, calibration, …) are preserved.
* A settings file that exists but cannot be parsed (a UTF-8 BOM is accepted; a
  trailing comma is not) is REPORTED, and nothing will write over it until it
  is fixed — the old behaviour read it as empty and the next save deleted
  every key the window does not manage.
* Writes use the atomic temp-file + ``os.replace`` pattern, so a crash
  mid-save can't corrupt the file.
* SECURITY: integration secrets are NEVER read from or written to the repo.
  The Integrations tab shows only presence for each env var and never a
  secret's value; the web-interface token field is masked. The three
  plain-text rows there (OBS_HOST_HINT / OBS_PORT_HINT / HUE_BRIDGE_IP_HINT)
  are OWNER NOTES: they persist to the user (gitignored) settings file and are
  read by NOTHING at runtime. Their labels and help say so. See the block
  comment above them for why they are neither wired nor deleted.
* ``tools/web_interface.py`` serves this module's ``SCHEMA`` as its settings
  panel, so a row added here appears there too.

Everything above the ``# ── GUI ──`` divider is import-safe with no GUI
dependency (and imports nothing from ``core`` at import time), so the tests can
exercise the schema, validation, load/save and the device/theme helpers on a
bare CI runner where tkinter is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid

# ──────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────
# tools/settings_window.py → project root is one level up.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Opened from the tray this file runs as a SCRIPT, so sys.path[0] is tools\ and
# the project root is NOT importable: the lazy `from core import …` below
# (VRAM budget, vision lockstep, config defaults) all failed, the VRAM panel
# read "unavailable" and a chat-model change stopped moving LOCAL_VISION_MODEL
# with it. Same fix as tools/jarvis_watchdog.py. Importing this module from the
# project (tests, web_interface) already has the root on the path — skip then.
if not any(os.path.normcase(os.path.abspath(p or os.curdir))
           == os.path.normcase(PROJECT_DIR) for p in sys.path):
    sys.path.insert(0, PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
# Default on-disk location of the live settings document. The actual path used
# by load/save is resolved at CALL time via ``settings_path()`` so a redirect
# (the ``JARVIS_SETTINGS_PATH`` env override below) takes effect even after this
# module is imported — e.g. the test runners point the whole suite at a
# throwaway file so a leaked ``save_settings`` can NEVER clobber the real one.
SETTINGS_PATH = os.path.join(DATA_DIR, "user_settings.json")
# Shipped, tracked template (data/ is fully gitignored).
EXAMPLE_PATH = os.path.join(PROJECT_DIR, "tools", "user_settings.example.json")
# The tray's command inbox, drained by the running JARVIS twice a second (see
# bobert_companion._drain_tray_commands_once). "Save & restart" appends
# {"cmd": "restart"} here exactly the way tray.py's Restart item does.
TRAY_COMMANDS_FILE = os.path.join(PROJECT_DIR, "tray_commands.json")

# Env var that redirects BOTH load and save away from ``SETTINGS_PATH``. When
# set and non-empty, every read/write (and the atomic temp file derived from it)
# uses this path instead. Unset/blank → the default above, i.e. today's
# behaviour exactly. Tests set this to a temp file; production never sets it.
SETTINGS_PATH_ENV = "JARVIS_SETTINGS_PATH"


def settings_path() -> str:
    """Resolve the settings file path, honouring the ``JARVIS_SETTINGS_PATH``
    override at call time. Returns that env var's value when set and non-empty,
    otherwise the staging copy for a STAGING-role process, otherwise the
    default ``data/user_settings.json``. Resolving here (rather than binding a
    module-level default once at import) is what lets a redirect set after
    import still take effect for both load and save.

    STAGING (2026-07-11): every settings-persisting action routes through this
    writer, but only JARVIS_SETTINGS_PATH used to redirect it — JARVIS_STAGING
    meant nothing here, so a staging harness (tools/action_smoke.py) that
    executed toggle actions wrote KINECT_GAZE_ENABLED / VOICE_CLONE_ENABLED
    straight into the LIVE prod file TWICE (2026-07-10 and again 2026-07-11,
    turning the face-tracker ci_sim gates red both times). A staging-role
    process now writes blue_green_manager's seeded copy in data_staging/."""
    override = (os.environ.get(SETTINGS_PATH_ENV) or "").strip()
    if override:
        return override
    if (os.environ.get("JARVIS_STAGING", "").strip() == "1"
            or "--staging" in sys.argv):
        return os.path.join(PROJECT_DIR, "data_staging", "user_settings.json")
    return SETTINGS_PATH

# Theme — identical palette to the tray dialogs.
BG = "#0d1117"
FG = "#c9d1d9"
FIELD_BG = "#161b22"
BORDER = "#30363d"
ACCENT = "#1f6feb"
MUTED = "#8b949e"
WARN = "#d29922"
ERROR = "#f85149"
OK_GREEN = "#3fb950"
FONT = ("Consolas", 10)
FONT_BOLD = ("Consolas", 11, "bold")
FONT_SECTION = ("Consolas", 10, "bold")
FONT_SMALL = ("Consolas", 9)

# The honest caveat. core/config.py applies data/user_settings.json ONCE, at
# import, when JARVIS starts; a few features also re-read their own keys, and
# some voice commands change a setting live — but the only thing that is true
# of EVERY row is "on the next start".
RESTART_NOTE = ("Saved settings take effect the next time JARVIS starts — "
                "use Save & restart.")

# Local Ollama endpoint + a STATIC fallback list of common chat tags, used to
# seed the "Local LLM model" dropdown until (or when) the background probe of
# the installed models answers. The live list is `installed_ollama_models()`.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_FALLBACK = [
    "gemma4:26b-a4b-it-qat",                # the default brain (core/config.py):
                                            # ~16 GB, multimodal — vision shares it
    "gemma4:12b",                           # game-mode brain: ~9 GB, multimodal
    "qwen2.5:14b-instruct-q5_K_M",          # proven text-only failover
    "qwen3:14b",                            # needs think:false (handled in-app)
    "llama3.1:8b-instruct-q5_K_M",
]
# The Claude models the "Claude model" and "Retry a failed turn on" rows
# offer: current models first, then the older ones (still selectable — the
# per-model request shaping in core.llm_client sends an older model exactly
# what it accepts). One list, so the two rows can never offer different sets.
CLAUDE_MODEL_CHOICES = [
    "claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5",
    "claude-fable-5-1", "claude-sonnet-5", "claude-sonnet-4-6",
    "claude-opus-4-8", "claude-opus-4-6",
]
# Tag substrings that are NOT chat models (embedding / vision) — excluded from
# the chat-model dropdown. Mirrors skills/model_picker's markers.
_NON_CHAT_MARKERS = (
    "nomic-embed", "embed-text", "-embed", "bge-", "all-minilm",
    "vl:", "-vl", "vision", "llava", "moondream", "bakllava",
)
# Embedding-only markers: the ONLY tags the vision-model dropdown leaves out
# (a dedicated VLM is a legitimate vision choice).
_EMBED_ONLY_MARKERS = ("nomic-embed", "embed-text", "-embed", "bge-",
                       "all-minilm")


def installed_ollama_models(base_url: str = OLLAMA_BASE_URL,
                            include_vision: bool = False) -> list[str]:
    """Installed Ollama tags via GET /api/tags, or the static
    OLLAMA_MODEL_FALLBACK list if Ollama is unreachable.

    ``include_vision=False`` (the chat dropdown) drops embedding AND vision
    models; ``True`` (the vision dropdown) drops embedding models only.
    Import-safe: ``requests`` is imported lazily so importing this module (for
    the tests / schema) never requires it or a network call. The GUI calls this
    on a background thread — a slow Ollama must not hold the window closed."""
    markers = _EMBED_ONLY_MARKERS if include_vision else _NON_CHAT_MARKERS
    try:
        import requests  # lazy: keep module import dependency-free
        r = requests.get(f"{base_url}/api/tags", timeout=(2, 3))
        if not r.ok:
            return list(OLLAMA_MODEL_FALLBACK)
        names = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
        keep = [n for n in names if not any(mk in n.lower() for mk in markers)]
        return keep or list(OLLAMA_MODEL_FALLBACK)
    except Exception:
        return list(OLLAMA_MODEL_FALLBACK)


# ──────────────────────────────────────────────────────────────────────────
#  Audio devices
# ──────────────────────────────────────────────────────────────────────────
# Synthetic mic-picker choices that don't map to a real device. They are part
# of the MICROPHONE_INDEX contract the monolith already honours (None = auto /
# PREFERRED_INPUT_DEVICES lookup; a NEGATIVE index = hard-off, no capture
# stream is opened — see bobert_companion._mic_input_disabled).
MIC_AUTO_LABEL = "System default (auto)"
MIC_OFF_LABEL = "Off (no mic)"
MIC_AUTO_INDEX = None
MIC_OFF_INDEX = -1


# -- legacy index helpers ---------------------------------------------------
# The picker used to persist the raw PortAudio INDEX, which renumbers whenever
# a USB device comes or goes (the same mic is #1 today and #4 tomorrow). The
# picker now saves the device NAME (see audio_device_choices below); these
# index helpers remain for an old file that still pins a number, and for the
# callers/tests that use them.
def list_input_devices() -> list[tuple[str, int]]:
    """Available audio INPUT devices as ``(label, index)`` tuples, e.g.
    ``("[2] Microphone (Realtek)", 2)``. Unfiltered, one row per PortAudio
    index. Import-safe and never raises; only the device list is read — no
    stream is opened."""
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
    """Legacy ``(label, index)`` list: auto, off, then every live input device;
    a saved index that is not present right now is kept visible."""
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
    """Resolve a stored MICROPHONE_INDEX to the legacy label to preselect."""
    for label, idx in choices:
        if idx == index:
            return label
    return MIC_AUTO_LABEL


def mic_label_to_index(label: str, choices: list[tuple[str, int]]):
    """Translate a legacy label back to the int (or None) it stands for."""
    for lbl, idx in choices:
        if lbl == label:
            return idx
    return MIC_AUTO_INDEX


# -- name-based picker (2026-09-30) -----------------------------------------
# Host-API names as PortAudio reports them on Windows → the short form shown.
_HOSTAPI_SHORT = {
    "MME": "MME",
    "Windows DirectSound": "DirectSound",
    "Windows WASAPI": "WASAPI",
    "Windows WDM-KS": "WDM-KS",
}
# Not devices: the aliases PortAudio adds on top of the real endpoints.
_ALIAS_DEVICE_MARKERS = ("sound mapper", "primary sound capture driver",
                         "primary sound driver")
# Loopback / virtual endpoints nobody means when they pick "my microphone".
_VIRTUAL_DEVICE_MARKERS = ("steam streaming", "vb-audio", "cable output",
                           "cable input", "voicemeeter", "virtual")
# Input endpoints that are sockets, not microphones.
_NOT_A_MIC_PREFIXES = ("line", "analog connector", "spdif", "s/pdif",
                       "stereo mix", "what u hear", "wave out mix",
                       "digital input")
# MME (WAVEINCAPS/WAVEOUTCAPS szPname[32]) truncates names to 31 characters, so
# the MME row of a long-named device is a PREFIX of its DirectSound/WASAPI row.
_MME_NAME_LIMIT = 31

DIRECTION_KEYS = {
    # direction -> (index key, preferred-names key)
    "input": ("MICROPHONE_INDEX", "PREFERRED_INPUT_DEVICES"),
    "output": ("SPEAKER_INDEX", "PREFERRED_OUTPUT_DEVICES"),
}


def _norm_device_name(name: str) -> str:
    n = " ".join(str(name or "").split()).lower()
    n = re.sub(r"\(\s+", "(", n)
    return re.sub(r"\s+\)", ")", n)


def is_real_audio_device(name: str, direction: str = "input") -> bool:
    """False for the rows a person never means: blank names, "Microphone ()"
    (an endpoint with nothing behind it), the Sound Mapper / Primary Sound
    Driver aliases, loopback/virtual devices, and — for inputs — line-level
    sockets ("Line", "Analog Connector", "Stereo Mix")."""
    n = _norm_device_name(name)
    if not n or re.search(r"\(\s*\)", n):
        return False
    if any(m in n for m in _ALIAS_DEVICE_MARKERS):
        return False
    if any(m in n for m in _VIRTUAL_DEVICE_MARKERS):
        return False
    if direction == "input" and n.startswith(_NOT_A_MIC_PREFIXES):
        return False
    return True


def _query_audio_devices():
    """(devices, hostapis) from sounddevice, or ([], []) when unavailable.
    Reads the device LIST only — never opens a stream. Never raises."""
    try:
        import sounddevice as sd  # lazy: PortAudio dependency, GUI-only
        return list(sd.query_devices()), list(sd.query_hostapis())
    except Exception:
        return [], []


def list_audio_devices(direction: str = "input", devices=None,
                       hostapis=None) -> list[dict]:
    """The real audio devices for ``direction`` ("input" | "output"), ONE row
    per physical device however many host APIs expose it.

    Each row: ``{"name", "label", "apis", "indices", "names"}``.
      * ``name`` is what the picker SAVES — the raw name of the device's
        lowest-index row (MME on Windows, whose 31-character truncation is a
        prefix of the other APIs' full name). bobert_companion._pick_device
        matches PREFERRED_*_DEVICES entries as case-insensitive SUBSTRINGS in
        index order, so this name finds the same device under every API.
      * ``label`` is what the owner sees: the full name plus the host APIs.

    ``devices``/``hostapis`` default to a live sounddevice query (the device
    list only; no stream is opened). Never raises."""
    if devices is None:
        devices, queried_apis = _query_audio_devices()
        if hostapis is None:
            hostapis = queried_apis
    hostapis = list(hostapis or [])
    chan_key = "max_input_channels" if direction == "input" \
        else "max_output_channels"
    # group key -> [(index, raw name, short host-API name), ...]
    groups: dict[str, list] = {}
    order: list[str] = []
    for idx, dev in enumerate(devices or []):
        try:
            if int(dev.get(chan_key, 0) or 0) <= 0:
                continue
            raw = str(dev.get("name", "") or "")
        except (AttributeError, TypeError, ValueError):
            continue
        if not is_real_audio_device(raw, direction):
            continue
        api = ""
        try:
            api_i = int(dev.get("hostapi", -1))
            if 0 <= api_i < len(hostapis):
                api_name = str(hostapis[api_i].get("name", ""))
                api = _HOSTAPI_SHORT.get(api_name, api_name)
        except (AttributeError, TypeError, ValueError):
            api = ""
        key = _norm_device_name(raw)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((idx, raw, api))
    # Fold an MME-truncated group into the ONE full-named group it prefixes.
    for key in list(order):
        members = groups.get(key)
        if not members or not any(len(n.strip()) >= _MME_NAME_LIMIT - 1
                                  for _i, n, _a in members):
            continue
        longer = [k for k in order
                  if k != key and k in groups and len(k) > len(key)
                  and k.startswith(key)]
        if len(longer) == 1:
            groups[longer[0]] = members + groups[longer[0]]
            del groups[key]
            order.remove(key)
    out = []
    for key in order:
        members = sorted(groups[key])
        apis = list(dict.fromkeys(a for _i, _n, a in members if a))
        save_name = members[0][1].strip() or members[0][1]
        display = max((n for _i, n, _a in members),
                      key=lambda n: len(n.strip())).strip()
        if apis == ["WDM-KS"]:
            api_txt = "WDM-KS only — may not open"
        else:
            api_txt = " · ".join(apis) if apis else "unknown API"
        out.append({"name": save_name, "label": f"{display}  [{api_txt}]",
                    "apis": apis, "indices": [i for i, _n, _a in members],
                    "names": [n for _i, n, _a in members]})
    return out


def _auto_label(direction: str) -> str:
    what = "mic" if direction == "input" else "speakers"
    return f"Automatic — Windows default {what} (or the preferred list)"


def audio_device_choices(direction: str, saved_index, preferred,
                         devices=None, hostapis=None) -> tuple[list[dict], dict]:
    """The picker rows for ``direction`` and the row to preselect.

    Returns ``(choices, initial)``; every choice is
    ``{"label", "kind", "value"}`` with kind one of:
      "auto"  — MICROPHONE_INDEX/SPEAKER_INDEX None: the preferred-name list,
                else the Windows default (the owner's normal setup);
      "off"   — input only: MICROPHONE_INDEX -1, no capture stream at all;
      "name"  — a device, saved by NAME as the first PREFERRED_*_DEVICES entry
                (bobert_companion._pick_device's existing name match), so it
                survives the renumbering that broke the old index picker;
      "index" — an old file that still pins a raw index: shown so it
                round-trips, and replaced as soon as a device is picked.

    The picker OWNS the first entry of the preferred list: when the saved index
    is None and the list is non-empty, the device matching its first entry is
    preselected (a "(saved, not connected)" row when nothing matches)."""
    preferred = [str(p) for p in (preferred or []) if str(p).strip()]
    if devices is None:
        devices, queried_apis = _query_audio_devices()
        if hostapis is None:
            hostapis = queried_apis
    rows = list_audio_devices(direction, devices=devices, hostapis=hostapis)
    choices: list[dict] = [{"label": _auto_label(direction), "kind": "auto",
                            "value": None}]
    if direction == "input":
        choices.append({"label": "Off — no microphone", "kind": "off",
                        "value": MIC_OFF_INDEX})
    for r in rows:
        choices.append({"label": r["label"], "kind": "name",
                        "value": r["name"], "names": r["names"]})
    initial = choices[0]
    saved = saved_index
    if isinstance(saved, bool):
        saved = None
    if isinstance(saved, int) and saved < 0 and direction == "input":
        initial = choices[1]
    elif isinstance(saved, int) and saved >= 0:
        here = ""
        for r in rows:
            if saved in r["indices"]:
                here = r["names"][r["indices"].index(saved)].strip()
        if not here:
            try:
                here = str(devices[saved].get("name", "")).strip()
            except Exception:
                here = ""
        initial = {"label": f"Device #{saved} — pinned by number "
                            f"({here or 'not connected'}; renumbers)",
                   "kind": "index", "value": saved}
        choices.append(initial)
    elif preferred:
        head = preferred[0].strip()
        low = head.lower()
        match = None
        for c in choices:
            if c["kind"] == "name" and c["value"].strip().lower() == low:
                match = c
                break
        if match is None:
            for c in choices:
                if c["kind"] == "name" and any(
                        low in n.lower() for n in c.get("names", [])):
                    match = c
                    break
        if match is None:
            match = {"label": f"{head}  [saved — not connected]",
                     "kind": "name", "value": head, "names": [head]}
            choices.append(match)
        initial = match
    # Labels must be unique — the combobox hands back a label.
    seen: dict[str, int] = {}
    for c in choices:
        n = seen.get(c["label"], 0)
        seen[c["label"]] = n + 1
        if n:
            c["label"] = f"{c['label']} ({n + 1})"
    return choices, initial


def device_choice_index(choice: dict):
    """The MICROPHONE_INDEX / SPEAKER_INDEX value a picker row stands for."""
    kind = (choice or {}).get("kind")
    if kind in ("off", "index"):
        return choice.get("value")
    return None


def device_choice_list(choice: dict, owned_head, current_list) -> list[str]:
    """The PREFERRED_*_DEVICES list after picking ``choice``.

    ``owned_head`` is the name the picker currently owns at the head of the
    list (the device it showed or last put there), or None. That one entry is
    replaced; every other name the owner typed is kept in order. "auto" drops
    the owned entry (back to the rest of the list, else the Windows default);
    "off"/"index" leave the list alone."""
    cur = [str(p) for p in (current_list or []) if str(p).strip()]
    kind = (choice or {}).get("kind")
    if kind not in ("auto", "name"):
        return cur
    rest = cur
    if owned_head:
        low = str(owned_head).strip().lower()
        rest = [p for p in cur if p.strip().lower() != low]
    if kind == "auto":
        return rest
    name = str(choice.get("value") or "").strip()
    if not name:
        return rest
    return [name] + [p for p in rest if p.strip().lower() != name.lower()]


# Order matters — drives both the Notebook tab order and `--tab` resolution.
# The tray's Settings menu opens voice / ai / privacy / integrations /
# advanced; hearing and cameras are new (2026-09-30) and reachable by tab.
TAB_ORDER = ["voice", "hearing", "ai", "cameras", "privacy", "integrations",
             "advanced"]
TAB_LABELS = {
    "voice": "Voice",
    "hearing": "Hearing & Mic",
    "ai": "AI & Models",
    "cameras": "Cameras & Kinect",
    "privacy": "Privacy",
    "integrations": "Integrations",
    "advanced": "Advanced",
}

# ──────────────────────────────────────────────────────────────────────────
#  Settings schema
# ──────────────────────────────────────────────────────────────────────────
# A flat dict of JSON-key → field-spec. Each spec is a dict with:
#   tab      one of TAB_ORDER (the sub-headings and row order inside each tab
#            live in TAB_SECTIONS below, so this dict keeps its historical
#            order — tools/web_interface.py lists tabs in first-seen order)
#   label    human label for the control
#   type     "bool" | "enum" | "str" | "combo" | "int" | "float" | "text" |
#            "routing" | "device" (persisted) — or "status" / "view"
#            (read-only rows whose keys start with "_")
#   default  default value (matches core/config.py's current default)
#   help     (optional) hint shown under the control
#   choices  (enum/combo/routing) allowed / suggested string values
#   min/max  (int/float) inclusive range; min_exclusive=True makes min strict.
#            Numbers are non-negative unless a row says otherwise. forbid maps
#            a value to the reason it is refused.
#   nonblank (str/combo) an empty value is refused
#   secret   (str) masked in the GUI
#   suggest  (combo) "ollama" / "ollama-vision" / "monitors": where the live
#            suggestions come from
#   direction/names_key (device) "input"|"output" and the preferred-name list
#            the picker writes
#   secret_env (status rows) the OS env vars probed for PRESENT/not-set; their
#            VALUES are never read into a control
#
# Keeping the schema as plain data (no tkinter) lets the tests assert on
# defaults / coverage and lets `default_settings()` build the template file
# without ever importing the GUI. tools/web_interface.py renders this SCHEMA
# as its settings panel.
#
# This is a curated subset of core/config.py — hardware-pinning structures
# (CAMERAS, MONITORS, CONFIRM_KEYWORDS, robot IPs) are not editable here;
# CAMERAS is SHOWN read-only.
SCHEMA: dict[str, dict] = {
    # ── Voice / Audio ──────────────────────────────────────────────────
    "VOICE_MODE": {
        "tab": "voice", "label": "Voice pipeline", "type": "enum",
        "choices": ["turn_based", "realtime"], "default": "turn_based",
        "help": "realtime = low-latency streaming; it needs RealtimeSTT, "
                "RealtimeTTS and PyAudio, and falls back to turn_based "
                "without them.",
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
    "REQUIRE_WAKE_MODE": {
        "tab": "voice", "label": "Wake-word mode (manual, Alexa-style)",
        "type": "bool", "default": False,
        "help": "Require a leading 'JARVIS' on every command until turned "
                "off. Also toggled by voice ('wake word mode on/off') — "
                "handy when an external TV the media session can't see is "
                "playing.",
    },
    "FOLLOWUP_WINDOW_S": {
        "tab": "voice", "label": "Follow-ups without the wake word for (seconds)",
        "type": "float", "default": 0.0, "max": 600,
        "help": "In wake-word mode, after you say 'JARVIS', follow-ups within "
                "this many seconds need no wake word, and each one extends "
                "the window. 0 = strict (every command needs 'JARVIS'). A TV "
                "talking in the room can hold the window open. Applies on the "
                "next start.",
    },
    # Device dialogues (core/dialogue.py). DIALOGUE_LOST_HOLD_S got its row
    # once the monolith's _dialogue_session read it (at each dialogue's end);
    # until then nothing in the tree did, and a row for a dead constant is the
    # dead-toggle bug tests/test_settings_schema_wiring.py catches.
    "SKILL_ROUTES_ENABLED": {
        "tab": "voice", "label": "Let skills claim exact requests before the AI",
        "type": "bool", "default": True,
        "help": "A skill can recognise an exact request (e.g. 'talk to the "
                "<device> about pizza') and run its action directly instead "
                "of leaving the choice to the AI model. Off sends every "
                "request to the model. Applies on the next start.",
    },
    "DIALOGUE_ENABLED": {
        "tab": "voice", "label": "Scripted back-and-forth with talking devices",
        "type": "bool", "default": True,
        "help": "Let a skill that owns a talking device run a short scripted "
                "exchange between JARVIS and it. Off refuses every dialogue. "
                "Applies on the next start.",
    },
    "DIALOGUE_MAX_S": {
        "tab": "voice", "label": "Longest dialogue (seconds)", "type": "int",
        "default": 40, "min": 1, "max": 600,
        "help": "Hard cap on one dialogue's length. Applies on the next start.",
    },
    "DIALOGUE_STOP_LISTEN": {
        "tab": "voice", "label": "Listen for 'stop' after each device line",
        "type": "bool", "default": True,
        "help": "Off = no stop-listening after device lines (the tray, the "
                "wake word and the device itself still stop a dialogue). "
                "Applies on the next start.",
    },
    "DIALOGUE_BEAT_S": {
        "tab": "voice", "label": "Pause after each device line (seconds)",
        "type": "float", "default": 0.6, "max": 10,
        "help": "Comic timing, and the window in which your 'stop' is heard. "
                "Applies on the next start.",
    },
    "DIALOGUE_LOST_HOLD_S": {
        "tab": "voice",
        "label": "Stay quiet after a device stops answering (seconds)",
        "type": "float", "default": 12.0, "max": 120,
        "help": "When a dialogue ends because the device stopped answering "
                "(you are probably talking to it), JARVIS holds his own "
                "unprompted speech and ignores what the mic hears without "
                "the wake word for this long. Saying 'JARVIS' and typed "
                "commands always get through. 0 = no hold. Applies on the "
                "next start.",
    },
    "NIGHT_QUIET_ENABLED": {
        "tab": "voice", "label": "Quieter voice at night (by the clock)",
        "type": "bool", "default": True,
        "help": "From about 22:00 JARVIS gets quieter, slower and shorter "
                "because of the time alone: a hushed voice from 23:00, the "
                "late-night tone and one-sentence replies, a softer wake "
                "greeting, the 'we've been at this a while' nudge, other "
                "proactive lines held once you've been silent a while, "
                "remarks about the hour after 01:00, and night-owl mode "
                "switching itself on. Off = his voice, reply length and "
                "proactive speech at night are the same as in the daytime; "
                "'night owl on' and telling him you're tired still work. "
                "Applies on the next start.",
    },
    "NIGHT_OWL_AUTO": {
        "tab": "voice", "label": "Night-owl mode switches on by itself at 23:00",
        "type": "bool", "default": True,
        "help": "From 23:00 to 06:00: a voice about 15% quieter and a little "
                "slower, one-short-sentence replies, no 'thinking' filler, "
                "non-essential announcements (weather, news, banter) held and "
                "the overlay dimmed. Off = only when you say 'night owl on'. "
                "'Quieter voice at night' off also stops it. Applies on the "
                "next start.",
    },
    "SELF_ECHO_FILTER_ENABLED": {
        "tab": "hearing", "label": "Never answer his own voice",
        "type": "bool", "default": True,
        "help": "Ignore anything the mic hears while JARVIS is speaking (or "
                "just after), and anything that repeats a line he said "
                "moments ago. 'Stop' always gets through; typed commands are "
                "never checked. Applies on the next start.",
    },
    "SELF_ECHO_WINDOW_S": {
        "tab": "hearing", "label": "Remember his own lines for (seconds)",
        "type": "float", "default": 20.0, "max": 600,
        "help": "How long a line JARVIS spoke is remembered, so the mic "
                "hearing it again is ignored. Applies on the next start.",
    },
    "SELF_ECHO_TAIL_S": {
        "tab": "hearing", "label": "Ignore the mic just after he speaks (seconds)",
        "type": "float", "default": 0.8, "max": 10,
        "help": "Speech that starts this soon after one of his own lines "
                "ends (while the mic was already listening) counts as his "
                "echo. Applies on the next start.",
    },
    "NOISE_FILTER_ENABLED": {
        "tab": "hearing", "label": "Ignore noise heard as 'Bye.' / 'Thank you.'",
        "type": "bool", "default": True,
        "help": "Whisper turns room noise into 'Bye.', 'Thank you.' or 'You'. "
                "Ignore a transcript that is only one of those when it was "
                "barely louder than silence, Whisper doubts it, or nobody has "
                "talked to JARVIS for a while. A real reply ('thank you' right "
                "after he answers) is kept; typed commands are never checked. "
                "Applies on the next start.",
    },
    "MICROPHONE_INDEX": {
        "tab": "hearing", "label": "Microphone", "type": "device",
        "default": None,
        "direction": "input", "names_key": "PREFERRED_INPUT_DEVICES",
        "help": "'Automatic' follows the Windows default mic (or the preferred "
                "names below). Picking a device saves its NAME as the first "
                "preferred mic, so it survives devices being renumbered; "
                "'Off' disables capture entirely. The list is your real input "
                "devices (read when this window opens).",
    },
    "PREFERRED_INPUT_DEVICES": {
        "tab": "hearing", "label": "Preferred mic names (automatic mode)",
        "type": "text", "default": [],
        "help": "One device-name fragment per line, most-preferred first. "
                "JARVIS uses the first connected match that opens, and "
                "switches as you plug and unplug. The picker above edits the "
                "first line. Empty = the Windows default.",
    },
    "TTS_VOICE": {
        "tab": "voice", "label": "TTS voice", "type": "str",
        "default": "en-GB-RyanNeural", "nonblank": True,
        "help": "Edge neural voice name, e.g. en-GB-RyanNeural.",
    },
    "TTS_BACKEND": {
        "tab": "voice", "label": "TTS backend", "type": "enum",
        "choices": ["edge", "kokoro", "pyttsx3", "xtts"], "default": "edge",
        "help": "edge = online neural; kokoro = local CPU (offline, frees the GPU); "
                "pyttsx3 = offline SAPI; xtts = Coqui voice clone (needs the "
                "TTS package). A backend that can't run falls back to another "
                "voice.",
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
        "tab": "hearing", "label": "Audio processing (master)", "type": "bool",
        "default": True,
        "help": "Master switch for the mic-cleanup chain below.",
    },
    "AUDIO_ECHO_CANCEL": {
        "tab": "hearing", "label": "Echo cancellation (AEC)", "type": "bool",
        "default": True,
        "help": "Cancel JARVIS's own playback from the mic.",
    },
    "AUDIO_NOISE_SUPPRESS": {
        "tab": "hearing", "label": "Noise suppression (NS)", "type": "bool",
        "default": True, "help": "Suppress stationary background noise.",
    },
    "AUDIO_AGC": {
        "tab": "hearing", "label": "Auto gain control (AGC)", "type": "bool",
        "default": True, "help": "Normalise mic level before STT.",
    },
    "VAD_THRESHOLD": {
        "tab": "hearing", "label": "VAD threshold", "type": "float",
        "default": 0.008, "min": 0, "min_exclusive": True, "max": 1,
        "help": "Mic RMS to treat as speech; raise to ignore more noise "
                "(typical 0.005-0.05).",
    },
    "AUDIO_DUCKING_ENABLED": {
        "tab": "hearing", "label": "Duck other apps while speaking", "type": "bool",
        "default": True,
        "help": "Lower other apps' volume while JARVIS talks (Windows).",
    },
    # 2026-09-29 audio-device flap damping (core/audio_flap.py).
    "AUDIO_FLAP_WINDOW_S": {
        "tab": "hearing", "label": "Device flapping window (seconds)",
        "type": "float", "default": 300.0, "max": 86400,
        "help": "A mic or speaker that changes the flapping number of times "
                "within this many seconds is 'flapping': JARVIS says so once "
                "and stays quiet about audio devices until it has been steady "
                "for twice this long. Applies on the next start.",
    },
    "AUDIO_FLAP_THRESHOLD": {
        "tab": "hearing", "label": "Device changes that count as flapping",
        "type": "int", "default": 3, "max": 100,
        "help": "How many changes inside the window make a device 'flapping'. "
                "Below 2 turns flap detection off. Applies on the next start.",
    },
    "AUDIO_ANNOUNCE_MIN_GAP_S": {
        "tab": "hearing", "label": "Min seconds between device announcements",
        "type": "float", "default": 60.0, "max": 3600,
        "help": "At most one spoken audio-device announcement per this many "
                "seconds; a newer one replaces one still waiting. 0 turns the "
                "limit off. Applies on the next start.",
    },
    "AUDIO_REPICK_STABLE_S": {
        "tab": "hearing", "label": "Follow a new default mic after (seconds)",
        "type": "float", "default": 8.0, "max": 600,
        "help": "When Windows moves the default mic or speakers, JARVIS "
                "follows once the new default has held this long -- at once "
                "if the device it is using has gone. 0 = follow immediately. "
                "Applies on the next start.",
    },
    # 2026-07-08: surface the Whisper STT device/model so the v2.0.23 crash-
    # workaround is settable AND persisted — previously they lived only in
    # core/config.py, so a Settings save (which rewrites user_settings.json from
    # the schema) dropped any hand-set WHISPER_DEVICE. Default stays 'auto'
    # (v2.0.23 made auto crash-safe via the VRAM plan); this is persistence
    # plumbing only and does not touch the runtime whisper code.
    "WHISPER_DEVICE": {
        "tab": "hearing", "label": "Whisper STT device", "type": "enum",
        "choices": ["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
        "default": "auto",
        "help": "Where speech-to-text runs. auto / cuda / cuda:0 = the main "
                "GPU; cuda:1 = the second card (keeps the main one free — the "
                "VRAM budget then leaves Whisper off the main card); cpu = no "
                "GPU. Applies on the next start.",
    },
    "WHISPER_MODEL_CUDA": {
        "tab": "hearing", "label": "Whisper GPU model", "type": "str",
        "default": "large-v3-turbo", "nonblank": True,
        "help": "faster-whisper model used on the GPU (large-v3-turbo: ~1.5 "
                "GB VRAM on the card chosen above, ~8x faster than large-v3 "
                "at near-identical accuracy). Applies on the next start.",
    },
    "SPEAKER_INDEX": {
        "tab": "hearing", "label": "Speakers", "type": "device",
        "default": None,
        "direction": "output", "names_key": "PREFERRED_OUTPUT_DEVICES",
        "help": "'Automatic' follows the Windows default speakers (or the "
                "preferred names below). Picking a device saves its NAME as "
                "the first preferred speaker.",
    },
    "PREFERRED_OUTPUT_DEVICES": {
        "tab": "hearing", "label": "Preferred speaker names (automatic mode)",
        "type": "text", "default": [],
        "help": "One device-name fragment per line, most-preferred first. "
                "JARVIS plays through the first connected match. The picker "
                "above edits the first line. Empty = the Windows default.",
    },
    # Headset auto-switch (audio/audio_switch.py). The config defaults come
    # from JARVIS_AUDIO_* env vars; the window shows the value in effect.
    "AUDIO_AUTOSWITCH_ENABLED": {
        "tab": "hearing", "label": "Move the default speakers with the headset's power",
        "type": "bool", "default": False,
        "help": "Watch the wireless headset's power (read from its dongle) and "
                "make it the Windows default speakers when it turns on, and "
                "put the previous default back when it turns off. Needs the "
                "headset name below. Applies on the next start.",
    },
    "AUDIO_AUTOSWITCH_HEADSET": {
        "tab": "hearing", "label": "Headset name (part of it)", "type": "str",
        "default": "",
        "help": "Part of the headset's Windows device name, e.g. CORSAIR VOID "
                "ELITE. List candidates with: python -m audio.audio_switch "
                "--list",
    },
    "AUDIO_AUTOSWITCH_FALLBACK": {
        "tab": "hearing", "label": "Speakers when the headset turns off",
        "type": "str", "default": "",
        "help": "Used when there is no earlier default to go back to (part of "
                "the device name).",
    },
    "AUDIO_AUTOSWITCH_MIC": {
        "tab": "hearing", "label": "Move the default MIC with the headset too",
        "type": "bool", "default": False,
        "help": "Off by default: a wrong move of the default recording device "
                "is how JARVIS goes deaf. Applies on the next start.",
    },
    "AUDIO_AUTOSWITCH_MIC_FALLBACK": {
        "tab": "hearing", "label": "Mic when the headset turns off",
        "type": "str", "default": "",
        "help": "Desk mic to fall back to (part of its name; check with "
                "python -m audio.audio_switch --list-mics). Blank = the mic is "
                "not moved, and JARVIS says so.",
    },
    "DEVICE_SPEECH_FILTER_ENABLED": {
        "tab": "hearing", "label": "Ignore what known talking devices say",
        "type": "bool", "default": True,
        "help": "A line that matches something a known device says (phrase "
                "lists in data/device_phrases/*.json) is ignored before it can "
                "wake JARVIS, be answered or be learned from. 'Stop' always "
                "gets through; no phrase files = nothing filtered. Applies on "
                "the next start.",
    },
    "MEDIA_VOICE_GATE_ENABLED": {
        "tab": "hearing", "label": "Ignore other voices while the PC plays audio",
        "type": "bool", "default": True,
        "help": "While a video, a reel or music plays on this PC, a 'JARVIS, "
                "...' in a voice that is clearly not yours is ignored (a reel "
                "once ran a command). 'Stop', pause / next / volume commands "
                "always get through, and an ignored 'JARVIS, ...' gets a "
                "short 'couldn't tell that was you'. Turn it off if it keeps "
                "missing you. Applies on the next start.",
    },
    "MEDIA_VOICE_GATE_REJECT_BELOW": {
        "tab": "hearing", "label": "Not-you voice score while audio plays",
        "type": "float", "default": 0.45, "min": 0, "max": 1,
        "help": "With the setting above on: while the PC plays audio, a voice "
                "scoring below this against your voiceprint is someone else "
                "and is ignored. Your own commands over a video have scored "
                "0.48-0.52, the reel 0.43. Lower it if he ignores you over "
                "your media; raise it if videos still set him off. Applies "
                "on the next start.",
    },
    # Double-clap trigger (skills/clap_trigger.py): read live by the skill's
    # worker / gate / routine; "clap trigger on/off" by voice saves the flag.
    "CLAP_TRIGGER_ENABLED": {
        "tab": "hearing", "label": "Double clap runs a routine",
        "type": "bool", "default": False,
        "help": "Two sharp claps, nothing else loud around them, run the clap "
                "routine below. Uses the microphone JARVIS already listens on; "
                "never while he is speaking, during the phone pings' quiet "
                "hours, in focus or game mode, or while music, a video or "
                "anything else plays on the speakers. 'Clap trigger on/off' "
                "by voice does the same and saves it here.",
    },
    "CLAP_TRIGGER_ACTION": {
        "tab": "hearing", "label": "Clap routine",
        "type": "enum", "default": "acknowledge",
        "choices": ["acknowledge", "predictive_morning_setup",
                    "morning_briefing"],
        "help": "acknowledge = just 'You rang, sir?' (start here: a "
                "mechanical key or a pen tap can pass for a clap, so listen "
                "for false triggers first). predictive_morning_setup = set up "
                "the workspace (Chrome and Apple Music, Teams, volume ~30%). "
                "morning_briefing = the briefing. A clap runs nothing else. "
                "'Clap trigger runs the morning setup' by voice sets it too.",
    },
    "CLAP_TRIGGER_WAKE": {
        "tab": "hearing", "label": "Clap to wake",
        "type": "bool", "default": False,
        "help": "A double clap while JARVIS is asleep / in standby wakes him "
                "and runs the routine (never during the quiet hours). Off = "
                "claps are ignored while asleep.",
    },
    "CLAP_TRIGGER_COOLDOWN_S": {
        "tab": "hearing", "label": "Clap routine cool-down (seconds)",
        "type": "float", "default": 60.0, "max": 3600,
        "help": "The routine runs at most once per this many seconds.",
    },
    "CLAP_TRIGGER_MIN_PEAK": {
        "tab": "hearing", "label": "How loud a clap must be (0-1)",
        "type": "float", "default": 0.12, "min": 0, "min_exclusive": True,
        "max": 1,
        "help": "Peak level of a clap at the microphone. 'Clap trigger status' "
                "tells you how loud your last clap was: lower this if your "
                "claps fall short, raise it if knocks across the room fire it.",
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
        "choices": list(CLAUDE_MODEL_CHOICES),
        "default": "claude-sonnet-5-5",
        "help": "Cloud model + est. cost PER CONVERSATION: Haiku 4.5 ~$0.02 "
                "(fastest), Sonnet 5.5 ~$0.04 (default — near-Opus smarts at "
                "Sonnet price), Opus 5.5 ~$0.08 (always thinks first: slower "
                "to start), Fable 5.1 ~$0.20 (the ceiling). Older models stay "
                "selectable. (Local Ollama is $0 — set the backend above to "
                "ollama.)",
    },
    "LOCAL_LLM_MODEL": {
        "tab": "ai", "label": "Local LLM model (Ollama, $0)", "type": "combo",
        "default": "gemma4:26b-a4b-it-qat", "nonblank": True,
        "suggest": "ollama",
        "choices": OLLAMA_MODEL_FALLBACK,
        "help": "Ollama tag for the always-on local brain — $0 per conversation. "
                "The list is your installed Ollama chat models (loaded in the "
                "background when this window opens); you can also type any "
                "tag, or switch by voice ('use the fast one').",
    },
    # 2026-09-30: a real row now. It used to be left out so a fresh install
    # wouldn't pin it — Save no longer pins untouched defaults, so that reason
    # is gone, and the vision brain deserves to be visible.
    "LOCAL_VISION_MODEL": {
        "tab": "ai", "label": "Local vision model", "type": "combo",
        "default": "gemma4:26b-a4b-it-qat", "nonblank": True,
        "suggest": "ollama-vision",
        "choices": OLLAMA_MODEL_FALLBACK + ["off"],
        "help": "The local model that looks at the screen. The SAME tag as the "
                "chat model means the one multimodal brain does both at no "
                "extra VRAM (the shipped setup); a different tag loads a "
                "second model; 'off' disables local vision. Changing the chat "
                "model moves this with it while the two are the same.",
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
        "help": "Retry a failed cloud vision call on the local vision model. "
                "When that is the chat brain (the shipped setup) it costs no "
                "extra VRAM; a separate vision model can load a second model "
                "— see the VRAM budget above.",
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
                "each turn. The embedder runs on the CPU by default "
                "(LTM_EMBED_DEVICE), so it uses no VRAM. Off = JARVIS "
                "remembers nothing new between sessions.",
    },
    "PROMPT_FREEZE_QUIET_S": {
        "tab": "ai", "label": "Hold prompt updates while talking (seconds)",
        "type": "float", "default": 30.0, "max": 3600,
        "help": "Local brain only: newly learned facts reach the prompt after "
                "this many seconds of quiet instead of between turns, so each "
                "reply starts warm (0 = update between turns). Applies on the "
                "next start.",
    },
    "LOCAL_PREFIX_REPRIME": {
        "tab": "ai", "label": "Re-warm the local brain after prompt updates",
        "type": "bool", "default": True,
        "help": "After a held-back prompt update, quietly send the new prompt "
                "to the already-loaded local model so your next turn is fast. "
                "Never loads a model, never in game mode or mid-turn. Applies "
                "on the next start.",
    },
    "FAST_PATHS_ENABLED": {
        "tab": "ai", "label": "Instant answers (dates, my name, last question)",
        "type": "bool", "default": True,
        "help": "Answer date questions ('how many days until Christmas'), "
                "'what did I just ask' and 'what's my name' instantly and "
                "exactly, without the AI model. Anything else still goes to "
                "the model. Applies on the next start.",
    },
    "INSTANT_ACTIONS_MODE": {
        "tab": "ai", "label": "Instant actions (volume, music, lights, print pause)",
        "type": "enum", "choices": ["shadow", "on", "off"],
        "default": "shadow",
        "help": "shadow = the AI model still answers, and JARVIS only logs "
                "the action it would have run instantly and whether the "
                "model agreed (data/instant_actions.jsonl: times and action "
                "names, never your words). on = 'volume up', 'pause the "
                "music', 'next song', 'turn off the lights', 'pause the "
                "print' run at once, without the model. off = neither. "
                "Questions and 'can you ...?' always go to the model. "
                "Applies on the next start.",
    },
    "TEAMS_NUDGE_ENABLED": {
        "tab": "ai", "label": "Teams unread-message nudger (background)",
        "type": "bool", "default": False,
        "help": "Every 10 minutes, read the screen with the vision model and "
                "say when Teams shows unread messages. Off by default; "
                "'check Teams' on request still works. Applies on the next "
                "start.",
    },
    "LOCAL_BACKGROUND_MAX_DEFER_S": {
        "tab": "ai", "label": "Hold background brain work while talking (max seconds)",
        "type": "float", "default": 120.0, "max": 3600,
        "help": "Local brain only: memory extraction, the ambient extractor "
                "and the Teams check wait until you go quiet, so they don't "
                "make your next reply re-read the whole prompt — but never "
                "longer than this. Your own requests never wait (0 = don't "
                "hold them). Applies on the next start.",
    },
    "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S": {
        "tab": "ai", "label": "Re-warm after background work (seconds since you spoke)",
        "type": "float", "default": 600.0, "max": 86400,
        "help": "After background work used the local brain, quietly re-warm "
                "your conversation if you spoke within this many seconds, so "
                "your next turn is fast. Same safeguards as the re-warm above "
                "(0 = off). Applies on the next start.",
    },
    "BACKGROUND_TAG_STRICT": {
        "tab": "ai", "label": "Also hold the newer background brain work",
        "type": "bool", "default": False,
        "help": "Local brain only: the notification sorter, Chappie, the "
                "credits check and the scheduled evening and morning "
                "briefings also wait while you are talking, like memory extraction does. Off: "
                "they run at once and the log notes when they would have "
                "waited. Applies on the next start.",
    },
    "LOCAL_REPRIME_AT_BOOT_S": {
        "tab": "ai", "label": "Warm the local brain after start (seconds)",
        "type": "float", "default": 20.0, "max": 3600,
        "help": "This many seconds after JARVIS starts, quietly send your "
                "conversation to the already-loaded local brain so your "
                "FIRST question is as fast as the rest. Same safeguards as "
                "the re-warm above (0 = off). Applies on the next start.",
    },
    "TURN_CHECK_MODE": {
        "tab": "ai", "label": "Check each turn for a failed reply",
        "type": "enum", "choices": ["off", "shadow", "on"],
        "default": "shadow",
        "help": "After each turn, check whether the reply claimed an "
                "action that never ran, ran nothing for a clear command, or "
                "named an action that does not exist. shadow = only log it "
                "(data/turn_check.jsonl holds kinds and action names, never "
                "your words); on = say 'One moment, sir.' and retry that "
                "turn once on Claude (a paid cloud call; needs Claude allowed "
                "for chat: backend claude, a key, chat not routed local); "
                "off = no check. Applies on the next start.",
    },
    "TURN_CHECK_ESCALATE_MODEL": {
        "tab": "ai", "label": "Retry a failed turn on", "type": "enum",
        "choices": list(CLAUDE_MODEL_CHOICES),
        "default": "claude-sonnet-5-5",
        "help": "The Claude model a failed turn is retried on when the "
                "check above is 'on'. Applies on the next start.",
    },
    "STREAMING_TTS_ENABLED": {
        "tab": "voice", "label": "Speak while replies stream", "type": "bool",
        "default": True,
        "help": "Start speaking the first sentence of a cloud reply while "
                "the rest is still generating (faster feel; action commands "
                "are never voiced early).",
    },
    "SENTENCE_TTS_ENABLED": {
        "tab": "voice", "label": "Start speaking after the first sentence",
        "type": "bool", "default": True,
        "help": "Kokoro voice only: a long reply starts playing its first "
                "sentence while the rest is still being rendered, instead of "
                "rendering the whole reply first. Short replies are voiced "
                "whole. Applies on the next start.",
    },
    "PROCESSING_FILLER_ENABLED": {
        "tab": "voice", "label": "Say 'just a moment' while thinking",
        "type": "bool", "default": False,
        "help": "Spoken turns only: a short butler line when a reply is slow. "
                "Never when muted, in standby, focus/DND, night-owl or game "
                "mode, or while the wake-word listener runs. Needs the Kokoro "
                "voice. Applies on the next start.",
    },
    "PROCESSING_FILLER_DELAY": {
        "tab": "voice", "label": "Filler delay (seconds)", "type": "float",
        "default": 2.5, "min": 0.5, "max": 60,
        "help": "Seconds of silence after you speak before 'Just a moment, "
                "sir.' (0.5-60). Applies on the next start.",
    },
    "PROCESSING_FILLER_STILL_DELAY": {
        "tab": "voice", "label": "'Still working' after (seconds of silence)",
        "type": "float", "default": 12.0, "max": 600,
        "help": "Seconds of silence in a long turn before one 'still working' "
                "line. Set at or below the filler delay to turn it off. "
                "Applies on the next start.",
    },
    "PROCESSING_FILLER_LATE_START_S": {
        "tab": "voice", "label": "Filler: latest start (seconds late)",
        "type": "float", "default": 3.0, "min": 0.1, "max": 60,
        "help": "How late the 'I heard you' line may still start when "
                "something holds it off. Only values under 1 change anything "
                "(a 1-second cap already applies); 0.6 is the speed plan's "
                "pick. Applies on the next start.",
    },
    "PROCESSING_FILLER_SKIP_PLEASANTRIES": {
        "tab": "voice", "label": "No filler for 'thank you' / 'hello'",
        "type": "bool", "default": False,
        "help": "Skip the filler line when all you said was a pleasantry "
                "(thank you, thanks, hello, hi, okay, good morning, good "
                "night, cool, great). Applies on the next start.",
    },
    "FILLER_DUCK_HOLD": {
        "tab": "voice", "label": "Keep music ducked from filler to answer",
        "type": "bool", "default": False,
        "help": "Music stays turned down from the filler line through the "
                "answer, instead of coming back up in between. Applies on the "
                "next start.",
    },
    "PROCESSING_FILLER_PRERENDER": {
        "tab": "voice", "label": "Prepare the answer during the filler",
        "type": "bool", "default": False,
        "help": "Kokoro voice only: while the filler line plays, start "
                "rendering the answer so it follows sooner. Used only when it "
                "sounds exactly like the normal answer would. Applies on the "
                "next start.",
    },
    "ANSWER_FIRST_ENABLED": {
        "tab": "voice", "label": "Answer first (skip 'one moment' lead-ins)",
        "type": "bool", "default": True,
        "help": "When a command speaks a real answer, skip the short "
                "'One moment, sir.' said before it, so the answer comes "
                "sooner. Lead-ins with numbers, questions or any real "
                "content are still spoken. Applies on the next start.",
    },
    "BARGE_IN_ENABLED": {
        "tab": "voice", "label": "Barge-in (interrupt him by voice)",
        "type": "bool", "default": True,
        "help": "Say the wake word while JARVIS is talking to cut him off "
                "and be heard. Requires the wake-word detector to be "
                "running — say 'JARVIS, start listening for the wake word' "
                "(wake_listener_start) first; his own voice can never "
                "trigger it.",
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
    # Until 2026-09-30 THIS row was labelled "Air control auto-start (Kinect
    # hand-mouse)" — but AIR_CONTROL_ENABLED is the other, dormant engine
    # (core/air_control.py); the live hand-mouse is KINECT_AIR_MOUSE_ENABLED
    # (skills/kinect_air_mouse.py), which had no row at all.
    "AIR_CONTROL_ENABLED": {
        "tab": "cameras",
        "label": "Air control — the OTHER hand engine (dormant): auto-start",
        "type": "bool", "default": False,
        "help": "NOT the air-mouse above: a separate, experimental engine "
                "(reach toward the sensor to take the cursor, fist to drag, "
                "point to scroll). Don't run it together with the air-mouse. "
                "On = start it at boot; 'air control on' by voice starts it "
                "for one session either way.",
    },
    "KINECT_ENABLED": {
        "tab": "cameras", "label": "Kinect sensor (master switch)",
        "type": "bool", "default": False,
        "help": "Let JARVIS open the Kinect v2 (a camera + microphone array "
                "pointed at the room). Every Kinect feature needs this. Runs "
                "on CPU/USB — no extra VRAM.",
    },
    "KINECT_AS_CAMERA": {
        "tab": "cameras", "label": "Use the Kinect as a face-tracking camera",
        "type": "bool", "default": False,
        "help": "Track faces on the Kinect's 1080p colour stream instead of "
                "the USB webcams.",
    },
    "KINECT_PRESENCE_ENABLED": {
        "tab": "cameras", "label": "Room presence from the skeleton",
        "type": "bool", "default": False,
        "help": "Count people and read head direction from the skeleton "
                "stream (better than the webcam guesswork).",
    },
    "KINECT_PRESENCE_STANDBY": {
        "tab": "cameras", "label": "Standby when the room is empty",
        "type": "bool", "default": False,
        "help": "Needs room presence. Off so the sensor never silences JARVIS "
                "unless you ask it to.",
    },
    "KINECT_PRESENCE_WAKE": {
        "tab": "cameras", "label": "Wake when someone walks in",
        "type": "bool", "default": False, "help": "Needs room presence.",
    },
    "KINECT_GREET_ON_ENTRY": {
        "tab": "cameras", "label": "Greet you when you come back",
        "type": "bool", "default": False,
        "help": "Needs room presence. A short greeting when you enter a room "
                "that was empty for a while — at most once a minute, never "
                "mid-conversation.",
    },
    "KINECT_POSTURE_NUDGE": {
        "tab": "cameras", "label": "Posture / stand-up nudges",
        "type": "bool", "default": False,
        "help": "Needs room presence. One gentle nudge after a long hunch or a "
                "long seated stretch, then a cool-down. Never nags.",
    },
    "KINECT_GAZE_ENABLED": {
        "tab": "cameras",
        "label": "Which monitor you're looking at (head direction)",
        "type": "bool", "default": False,
        "help": "The Kinect becomes the main 'which monitor' signal and works "
                "with the webcams off; the webcam guess is the fallback. Tune "
                "it with 'calibrate gaze'.",
    },
    "KINECT_GESTURES_ENABLED": {
        "tab": "cameras", "label": "Gestures (wave, swipe)",
        "type": "bool", "default": False,
        "help": "A wave wakes JARVIS from standby, a swipe cancels. A raised "
                "hand never confirms - a pending action needs a spoken 'yes'.",
    },
    "KINECT_POINT_CONTROL_ENABLED": {
        "tab": "cameras", "label": "Point at a device to control it",
        "type": "bool", "default": False,
        "help": "Point at a calibrated lamp or fan and say 'turn that on'. "
                "Calibrate each one with 'calibrate pointing for the desk "
                "lamp' while pointing at it.",
    },
    "KINECT_GUARD_ENABLED": {
        "tab": "cameras", "label": "Allow guard mode",
        "type": "bool", "default": False,
        "help": "Lets you arm guard mode by voice ('guard the room'): it "
                "watches every camera and alerts once on motion. This only "
                "allows arming — it never arms itself.",
    },
    "KINECT_SKELETON_OVERLAY_ENABLED": {
        "tab": "cameras", "label": "Skeleton overlay in the HUD camera tile",
        "type": "bool", "default": False,
        "help": "Show the Kinect colour image with the tracked skeleton drawn "
                "on it in the HUD preview.",
    },
    "KINECT_AIR_MOUSE_ENABLED": {
        "tab": "cameras",
        "label": "Kinect air-mouse (hand-mouse): raise a hand to take the cursor",
        "type": "bool", "default": False,
        "help": "Raise a hand above the shoulder to take the cursor; close the "
                "left or right hand to click that button, hold it closed to "
                "drag; lower the hand to let go. 'Air mouse on/off' by voice "
                "does the same and saves it here.",
    },
    "AIR_MOUSE_REQUIRE_OPEN_PALM": {
        "tab": "cameras", "label": "Air-mouse: require an open palm to engage",
        "type": "bool", "default": True,
        "help": "Passive mode only takes the cursor on an OPEN palm raised + held "
                "briefly (fewer false triggers from a closed/pointing hand "
                "reaching or gesturing). Off = height alone can engage.",
    },
    "AIR_MOUSE_ARM_RELAXES_GATE": {
        "tab": "cameras", "label": "Air-mouse: 'take the cursor' relaxes the gate",
        "type": "bool", "default": True,
        "help": "When you say 'take the cursor' / 'mouse control on' the strict "
                "smart-pose gate relaxes to height-only so a raised hand engages "
                "right away. Off = still needs the full open-palm hold even armed.",
    },
    "AIR_MOUSE_FIST_RELEASES": {
        "tab": "cameras", "label": "Air-mouse: a held fist releases the cursor",
        "type": "bool", "default": False,
        "help": "When ON, holding a closed fist for ~0.6 s lets go of the cursor "
                "without lowering your hand. OFF by default because it fights the "
                "click/drag gesture (a normal close stopped tracking). With it off, "
                "close to click/drag and lower your hand to let go.",
    },
    "AIR_MOUSE_PER_APP_DISABLE": {
        "tab": "cameras", "label": "Air-mouse: stand down over fullscreen games/video",
        "type": "bool", "default": True,
        "help": "Automatically disable the air-mouse when a fullscreen game or "
                "video player is in the foreground (edit the app list in "
                "data/user_settings.json → AIR_MOUSE_DISABLED_APP_HINTS).",
    },
    "KINECT_TWO_HAND_ENABLED": {
        "tab": "cameras", "label": "Two-hand move / resize of the front window",
        "type": "bool", "default": True,
        "help": "Raise both hands and grab to move the foreground window; "
                "spread or pinch to resize it. The one-hand cursor stands down "
                "meanwhile.",
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
    "GAME_MODE_ENABLED": {
        "tab": "ai", "label": "Game mode watcher (smaller brain while gaming)",
        "type": "bool", "default": False,
        "help": "Start the game watcher at boot: when a listed game holds the "
                "foreground, JARVIS swaps to the smaller game brain and pauses "
                "the camera/gesture extras until the game exits. Off = only by "
                "voice ('game mode on', 'low power mode'). Applies on the next "
                "start.",
    },

    # ── Privacy / Ambient ──────────────────────────────────────────────
    "AMBIENT_LISTEN_ENABLED": {
        "tab": "privacy", "label": "Ambient listening", "type": "bool",
        "default": False,
        "help": "Passively transcribe surroundings to learn context. OFF "
                "by default.",
    },
    "LEARN_ONLY_FROM_OWNER": {
        "tab": "privacy", "label": "Learn only from me", "type": "bool",
        "default": False,
        "help": "Learn facts only from what you type, say after 'JARVIS', say "
                "in your enrolled voice, or say in a conversation you started. "
                "Other people in the room never teach him. Without an "
                "enrolled voiceprint, only typed and 'JARVIS' turns teach. "
                "Applies on the next start.",
    },
    "LEARN_FOLLOWUP_S": {
        "tab": "privacy", "label": "Your conversation lasts (seconds)",
        "type": "float", "default": 90.0, "min": 0, "max": 900,
        "help": "With 'Learn only from me' on: after a turn that was clearly "
                "yours, follow-ups for this long can teach too (0 = only the "
                "clearly-yours turns). Applies on the next start.",
    },
    "LEARN_VOICE_REJECT_BELOW": {
        "tab": "privacy", "label": "Not-you voice score below",
        "type": "float", "default": 0.60, "min": 0, "max": 1,
        "help": "With 'Learn only from me' on: a voice scoring below this "
                "against your voiceprint is someone else and never teaches, "
                "even after 'JARVIS' -- and that includes YOUR voice on a "
                "turn where it matches below this score, so keep it under "
                "your usual match score (voice ID names you from 0.72). "
                "Raise it if guests still teach him; lower it if your own "
                "turns are refused. Applies on the next start.",
    },
    "AMBIENT_SCREEN_ENABLED": {
        "tab": "privacy", "label": "Ambient screen capture", "type": "bool",
        "default": False,
        "help": "Periodically read the screen for ambient context.",
    },
    "STANDBY_LOOP_ENABLED": {
        "tab": "voice", "label": "Standby music auto-detect", "type": "bool",
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
    "FACE_ID_ENABLED": {
        "tab": "privacy", "label": "Face recognition (who is at the desk)",
        "type": "bool", "default": False,
        "help": "Recognise faces from the monitor webcams. Face data is "
                "biometric: it stays in data/face_enroll.json on this PC, and "
                "nothing is stored until you say 'learn my face'. Off = no "
                "face recognition at all.",
    },
    "GREET_NEW_PEOPLE_ENABLED": {
        "tab": "privacy", "label": "Say hello when new people arrive",
        "type": "bool", "default": False,
        "help": "When several faces JARVIS doesn't know appear, one short "
                "hello (at most every ~10 minutes, never mid-conversation). "
                "Needs face recognition. Voice: 'say hi to guests'.",
    },
    "DAILY_BUDGET_USD": {
        "tab": "ai", "label": "Daily Claude $ cap", "type": "float",
        "default": 1.0, "max": 1000,
        "help": "Soft daily ceiling for the Chappie continuous-learning loop's "
                "Claude API spend (USD).",
    },
    "DEEP_AUDIT_BUDGET_USD": {
        "tab": "ai", "label": "Deep-audit daily $ cap", "type": "float",
        "default": 5.0, "max": 1000,
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
        # Every one of these is OPTIONAL (skills/obs_control._config defaults
        # the host/port, and an OBS without auth needs no password), so the old
        # all-required "not set" read as broken on a working setup. What CAN be
        # missing is the client package.
        "secret_env": [],
        "optional_env": ["OBS_HOST", "OBS_PORT", "OBS_PASSWORD"],
        "requires_module": "obswebsocket", "module_pip": "obs-websocket-py",
        "defaults_note": "127.0.0.1:4455, no password",
        # Names the real source in the same tab, exactly as _status_hue does.
        # Its absence is why the note fields below could be mistaken for the
        # thing that configures OBS.
        "help": "Configured via OBS_HOST / OBS_PORT / OBS_PASSWORD in your "
                ".env or OS environment — all optional (defaults "
                "127.0.0.1:4455, no password).",
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
    # Phone pings (core/phone_ping.py): read live on every ping, so a change
    # here applies at once. "turn off / on phone pings" by voice saves
    # PHONE_PING_ENABLED.
    "PHONE_PING_ENABLED": {
        "tab": "integrations", "label": "Ping my phone when something needs me",
        "type": "bool", "default": True,
        "help": "Print, confirmation and robot pings and the summary. Does "
                "nothing until the phone bridge can message you (a Telegram "
                "token plus your TELEGRAM_USER_ID, or ntfy, or Pushover, in "
                ".env). Ask 'how do I connect my phone' for the steps. Guard "
                "alerts have their own switch below and ignore this one.",
    },
    "PHONE_PING_PRINT": {
        "tab": "integrations", "label": "Print finished / failed / paused",
        "type": "bool", "default": True,
        "help": "Bambu printer: a print finished, failed (not one you "
                "cancelled), or paused with an error — once per pause.",
    },
    "PHONE_PING_CONFIRM": {
        "tab": "integrations", "label": "Unanswered confirmation while away",
        "type": "bool", "default": False,
        "help": "Something YOU asked for waited on your 'yes' and you stepped "
                "away. It lapses after 45 seconds, so this is a heads-up that "
                "nothing ran, not a question. Only the action's name is sent, "
                "never its details.",
    },
    "PHONE_PING_SECURITY": {
        "tab": "integrations", "label": "Guard-mode alerts",
        "type": "bool", "default": True,
        "help": "Critical: sent even in quiet hours and focus mode. Its own "
                "switch: turning phone pings off above does not stop guard "
                "alerts; turn this off to stop them.",
    },
    "PHONE_PING_ROBOT": {
        "tab": "integrations", "label": "Robot needs attention",
        "type": "bool", "default": True,
        "help": "A robot skill reporting it needs you.",
    },
    "PHONE_PING_SUMMARY": {
        "tab": "integrations", "label": "Daily summary",
        "type": "bool", "default": False,
        "help": "One message a day at the time below: what happened since "
                "the last one, including anything quiet hours held back.",
    },
    "PHONE_PING_SUMMARY_TIME": {
        "tab": "integrations", "label": "Summary time (HH:MM)",
        "type": "combo", "default": "07:30", "nonblank": True,
        "choices": ["07:00", "07:30", "08:00", "21:00", "22:00"],
        "help": "Local time. Morning (07:30) or nightly (22:00).",
    },
    "PHONE_PING_MAX_PER_HOUR": {
        "tab": "integrations", "label": "Most pings in an hour",
        "type": "int", "default": 6, "max": 60,
        "help": "Ordinary pings in any rolling hour; security alerts don't "
                "count.",
    },
    "PHONE_PING_QUIET_START": {
        "tab": "integrations", "label": "Quiet hours start (HH:MM)",
        "type": "combo", "default": "23:00", "nonblank": True,
        "choices": ["21:00", "22:00", "23:00", "00:00"],
        "help": "Ordinary pings wait until quiet hours end (or you wake "
                "JARVIS from overnight mode), then arrive as one message. The "
                "double-clap trigger ignores claps in these hours too. Same "
                "start and end = no quiet hours.",
    },
    "PHONE_PING_QUIET_END": {
        "tab": "integrations", "label": "Quiet hours end (HH:MM)",
        "type": "combo", "default": "07:00", "nonblank": True,
        "choices": ["06:00", "07:00", "08:00", "09:00"],
        "help": "Local time.",
    },
    "PHONE_PING_AWAY_MIN": {
        "tab": "integrations", "label": "Only ping after this long away (min)",
        "type": "float", "default": 10.0, "max": 240,
        "help": "Minutes since you last said something to JARVIS — or, while "
                "he is awake and talking, typed or moved the mouse. While you "
                "are here he tells you out loud instead. 0 = ping anyway.",
    },
    "PHONE_PING_CONFIRM_AFTER_MIN": {
        "tab": "integrations", "label": "Unanswered confirmation: wait (min)",
        "type": "float", "default": 2.0, "max": 240,
        "help": "How old an unanswered confirmation must be before it pings.",
    },
    # Moved from the AI tab, where it had nothing to do with models (P1-11).
    "STREAMING_AUTO_FULLSCREEN": {
        "tab": "integrations", "label": "Auto-fullscreen TV shows & movies",
        "type": "bool", "default": True,
        "help": "After a show or movie actually starts playing, send the "
                "player fullscreen ('f' on YouTube / Netflix / Disney+ / "
                "Prime / Hulu / Max). Off = playback starts windowed.",
    },
    # ── OWNER NOTES — persisted, but READ BY NOTHING ────────────────────
    # These three rows are a notepad, not configuration. Verified 2026-08-20:
    # a whole-tree grep for the names finds this schema, the example settings
    # file, two explanatory comments in tools/web_interface.py, the tests, and
    # the persisted (empty) values — and ZERO runtime readers.
    # core/config.py:1252 guarantees it stays that way: _apply_user_settings
    # skips any key with no module-level constant, and
    # tests/test_settings_schema_wiring.py asserts none of these ever gains one.
    #
    # They are NOT wired, deliberately:
    #   * HUE_BRIDGE_IP_HINT — skills/sh_hue.py already owns `bridge_ip` as a
    #     SINGLE authority with a self-heal that REWRITES the stored address
    #     when a configured one fails. A second source for the same field is
    #     this repo's #1 bug class: the GUI note and the self-healed config file
    #     would silently disagree the moment discovery corrected a stale IP.
    #   * OBS_*_HINT — skills/obs_control._config() reads OBS_HOST / OBS_PORT
    #     from the ENVIRONMENT on every call so they can change without a
    #     restart. Routing them through a restart-scoped settings file instead
    #     would be a downgrade, and its failure line already NAMES the endpoint
    #     it dialled ("I couldn't reach OBS at 127.0.0.1:4455, sir"), so a
    #     mismatch is audible rather than silent.
    # They are NOT deleted either: the round-trip path for schema keys with no
    # core.config constant is real machinery (tools/web_interface's _NO_CONSTANT
    # sourcing, pinned by tests/test_web_interface.GuiOnlyKeyRoundTripTests) and
    # these are its only live instances.
    #
    # What WAS wrong is that they read as live knobs. The old help split the
    # password (env) from the host (this box) — the exact inverse of the truth,
    # since OBS_HOST env is the only thing obs_control reads. Labels and help
    # now say so outright. Keep it that way: anything here that sounds like
    # configuration is a lie the owner will troubleshoot against.
    "OBS_HOST_HINT": {
        "tab": "integrations", "label": "OBS host (note only — not read)",
        "type": "str", "default": "",
        "help": "Reminder field. JARVIS reads OBS_HOST from your .env / OS "
                "environment (default 127.0.0.1) — typing here records the "
                "value, it does NOT change what JARVIS dials.",
    },
    "OBS_PORT_HINT": {
        "tab": "integrations", "label": "OBS port (note only — not read)",
        "type": "str", "default": "",
        "help": "Reminder field. JARVIS reads OBS_PORT from your .env / OS "
                "environment (default 4455); this value is not read.",
    },
    "HUE_BRIDGE_IP_HINT": {
        "tab": "integrations", "label": "Hue bridge IP (note only — not read)",
        "type": "str", "default": "",
        "help": "Reminder field. The live bridge IP lives in "
                "data/sh_hue_config.json (auto-discovered, and self-healed "
                "when a stored address stops answering); this value is not "
                "read.",
    },

    # ── Advanced ───────────────────────────────────────────────────────
    "HUD_ENABLED": {
        "tab": "advanced", "label": "On-screen HUD", "type": "bool",
        "default": True, "help": "Drives the unified HUD at boot.",
    },
    "HUD_MONITOR": {
        "tab": "advanced", "label": "HUD monitor", "type": "combo",
        "default": "top", "nonblank": True, "suggest": "monitors",
        "choices": ["top", "left", "middle", "right"],
        "help": "Which monitor the HUD first appears on (a name from MONITORS "
                "in core/config.py). Once you drag the HUD, its saved position "
                "wins. Applies on the next start.",
    },
    # Brain glow (core/brain_glow.py): read live by every publish.
    "BRAIN_GLOW_ENABLED": {
        "tab": "advanced", "label": "HUD glow shows which brain answers",
        "type": "bool", "default": True,
        "help": "A glowing ring around the reactor takes the colour of the "
                "brain that is answering: local model blue, Claude Sonnet "
                "gold, Opus violet, Haiku teal. Changes when you switch "
                "brains and when a turn falls back to the other brain. The "
                "halo keeps the listening / thinking / speaking colours; no "
                "ring while JARVIS is asleep.",
    },
    "BRAIN_GLOW_LABEL_S": {
        "tab": "advanced", "label": "Brain name label (seconds)",
        "type": "float", "default": 4.0, "min": 0, "max": 60,
        "help": "How long the brain's name shows under the reactor after it "
                "changes. 0 = colour only, no label.",
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
    # ── Cameras (read-only view) ────────────────────────────────────────
    # CAMERAS is a hardware-pinning structure (see the SCHEMA comment), so it
    # is SHOWN, not edited. A "_" key: never persisted, never in the web panel.
    "_view_cameras": {
        "tab": "cameras", "label": "Cameras (read-only)", "type": "view",
        "source_key": "CAMERAS",
        "help": "Edit these in user_settings.json (CAMERAS) — use 'Open "
                "user_settings.json' below. A camera with a name is found by "
                "that name, so USB re-plugs can't swap them; the index is only "
                "the fallback. Applies on the next start.",
    },
    # ── Camera open gate (core/camera_gate.py, 2026-09-29) ──────────────
    "CAMERA_REOPEN_MAX_BACKOFF_S": {
        "tab": "cameras", "label": "Camera retry ceiling (seconds)",
        "type": "float", "default": 600.0, "max": 86400,
        "help": "After a camera fails to open or keeps dropping frames, JARVIS "
                "waits 30 s before retrying, then 60, 120, 300, 600 and "
                "doubling — never longer than this ceiling — and resets after "
                "a minute of healthy video. 0 = retry without waiting. Applies "
                "on the next start.",
    },
    "USB_STORM_COOLDOWN_S": {
        "tab": "cameras", "label": "USB trouble: leave cameras alone for (seconds)",
        "type": "float", "default": 600.0, "max": 86400,
        "help": "When several cameras (or a camera and an audio device) drop "
                "at once, JARVIS stops opening every camera and the Kinect for "
                "this long, doubling if it happens again within the hour (max "
                "an hour). Running streams are left alone. 0 = off. Applies on "
                "the next start.",
    },
    "CAMERA_STORM_PROBATION_S": {
        "tab": "cameras", "label": "USB trouble: probation after a cool-down (seconds)",
        "type": "float", "default": 180.0, "max": 86400,
        "help": "For this long after the cameras are allowed back (and for a "
                "minute after any camera is reopened while the USB bus has "
                "been unstable within the hour), a single camera dropping "
                "stops every camera again, for twice as long. 0 = off. "
                "Applies on the next start.",
    },
    "CAMERA_CULPRIT_WINDOW_S": {
        "tab": "cameras", "label": "USB trouble: blame a camera started this recently (seconds)",
        "type": "float", "default": 5.0, "max": 600,
        "help": "When the USB bus drops out within this many seconds of one "
                "camera starting its video, that camera gets the blame for "
                "it. 0 = never blame a camera. Applies on the next start.",
    },
    "CAMERA_CULPRIT_THRESHOLD": {
        "tab": "cameras", "label": "USB trouble: switch a camera off after this many strikes",
        "type": "int", "default": 2, "max": 100,
        "help": "A camera blamed this many times within an hour is switched "
                "off for the rest of the session (JARVIS tells you once; say "
                "'use the left webcam again' after moving it to another "
                "port). 0 = never switch one off. Applies on the next start.",
    },
    "CAMERA_DIES_ON_OPEN_RETRY_S": {
        "tab": "cameras", "label": "Camera that drops out on every start: retry every (seconds)",
        "type": "float", "default": 1800.0, "max": 86400,
        "help": "A camera or the Kinect whose video dies within seconds of "
                "each of three starts in a row (it drops off USB as soon as "
                "it streams, usually a power problem) is retried only this "
                "often, doubling to at most an hour, instead of every 10 "
                "minutes. JARVIS tells you once; 'use the Kinect again' "
                "retries it now. 0 = off. Applies on the next start.",
    },
    "CAMERA_OPEN_MIN_GAP_S": {
        "tab": "cameras", "label": "Gap between two parts opening one camera (seconds)",
        "type": "float", "default": 10.0, "max": 600,
        "help": "The face tracker, start-up checks, self-diagnostic, preview "
                "tiles and Kinect never open the same device closer together "
                "than this. 0 = off. Applies on the next start.",
    },
    "TV_DETECT_ENABLED": {
        "tab": "cameras", "label": "Notice a lit TV (stops learning from TV chatter)",
        "type": "bool", "default": False,
        "help": "Look for a bright, flickering screen in the camera frame the "
                "face tracker already has, and treat it as media playing so "
                "JARVIS doesn't learn from what the TV says. It can only stop "
                "learning, never trigger anything. Calibrate with 'calibrate "
                "the tv region'; also 'turn on/off tv detection'.",
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
        "default": 8766, "min": 1, "max": 65535,
        "forbid": {8443: "8443 is the AirTag tracker's port"},
        "help": "TCP port for the dashboard (do not use 8443 — that's the "
                "AirTag tracker).",
    },
    "WEB_INTERFACE_BIND": {
        "tab": "advanced", "label": "Web interface bind address", "type": "str",
        "default": "127.0.0.1", "nonblank": True,
        "help": "127.0.0.1 = localhost only (safe, nothing off-box can reach "
                "it). 0.0.0.0 or a LAN IP EXPOSES it to your whole network and "
                "REQUIRES a token below — the server refuses to start otherwise.",
    },
    "WEB_INTERFACE_TOKEN": {
        "tab": "advanced", "label": "Web interface token", "type": "str",
        "default": "", "secret": True,
        "help": "Shared secret required on every request when set (and MANDATORY "
                "for a non-localhost bind). Treat it like a password — anyone "
                "with it can command JARVIS from any device on your LAN.",
    },
    "DASHBOARD_SHOW_TRANSCRIPTS": {
        "tab": "advanced", "label": "Show what was said in the web timeline",
        "type": "bool", "default": False,
        "help": "The dashboard's 'What JARVIS did' timeline shows each turn's "
                "words only when this is on AND the browser is on this PC — "
                "never to another device on your network. Applies on the next "
                "start.",
    },
}

# Field types whose key is a real persisted setting (everything except the
# read-only "status" / "view" rows, whose keys start with "_").
_PERSISTED_TYPES = {"bool", "enum", "str", "combo", "int", "float", "text",
                    "routing", "device"}


def persisted_keys() -> list[str]:
    """The schema keys that map to a stored value (excludes status/view rows)."""
    return [k for k, s in SCHEMA.items() if s.get("type") in _PERSISTED_TYPES]


# The window's layout: per tab, (sub-heading, rows in display order). Every
# SCHEMA row appears exactly once, under its own tab (pinned by the tests); a
# row missing here would still render, under "Other" at the end of its tab.
TAB_SECTIONS: dict[str, list[tuple[str, list[str]]]] = {
    "voice": [
        ("Speaking", ["TTS_BACKEND", "TTS_VOICE", "STREAMING_TTS_ENABLED",
                      "SENTENCE_TTS_ENABLED", "ANSWER_FIRST_ENABLED",
                      "PROCESSING_FILLER_ENABLED", "PROCESSING_FILLER_DELAY",
                      "PROCESSING_FILLER_STILL_DELAY",
                      "PROCESSING_FILLER_LATE_START_S",
                      "PROCESSING_FILLER_SKIP_PLEASANTRIES",
                      "FILLER_DUCK_HOLD", "PROCESSING_FILLER_PRERENDER"]),
        ("Voice clone", ["VOICE_CLONE_ENABLED", "VOICE_CLONE_PROFILE",
                         "VOICE_CLONE_MODEL"]),
        ("Wake word & conversation", [
            "VOICE_MODE", "WAKE_WORD_AUTOSTART", "START_IN_STANDBY",
            "REQUIRE_WAKE_MODE", "FOLLOWUP_WINDOW_S",
            "AMBIENT_MUSIC_REFUSE_WAKE", "STANDBY_LOOP_ENABLED",
            "BARGE_IN_ENABLED", "FOCUS_MODE_ENABLED"]),
        ("Device dialogues", ["SKILL_ROUTES_ENABLED", "DIALOGUE_ENABLED",
                              "DIALOGUE_MAX_S", "DIALOGUE_STOP_LISTEN",
                              "DIALOGUE_BEAT_S", "DIALOGUE_LOST_HOLD_S"]),
        ("At night", ["NIGHT_QUIET_ENABLED", "NIGHT_OWL_AUTO"]),
    ],
    "hearing": [
        ("Microphone", ["MICROPHONE_INDEX", "PREFERRED_INPUT_DEVICES",
                        "VAD_THRESHOLD", "AUDIO_PROCESSING_ENABLED",
                        "AUDIO_ECHO_CANCEL", "AUDIO_NOISE_SUPPRESS",
                        "AUDIO_AGC"]),
        ("Speakers", ["SPEAKER_INDEX", "PREFERRED_OUTPUT_DEVICES",
                      "AUDIO_DUCKING_ENABLED"]),
        ("Headset auto-switch", ["AUDIO_AUTOSWITCH_ENABLED",
                                 "AUDIO_AUTOSWITCH_HEADSET",
                                 "AUDIO_AUTOSWITCH_FALLBACK",
                                 "AUDIO_AUTOSWITCH_MIC",
                                 "AUDIO_AUTOSWITCH_MIC_FALLBACK"]),
        ("Device changes", ["AUDIO_FLAP_WINDOW_S", "AUDIO_FLAP_THRESHOLD",
                            "AUDIO_ANNOUNCE_MIN_GAP_S",
                            "AUDIO_REPICK_STABLE_S"]),
        ("Speech recognition", ["WHISPER_DEVICE", "WHISPER_MODEL_CUDA"]),
        ("What he ignores", ["SELF_ECHO_FILTER_ENABLED", "SELF_ECHO_WINDOW_S",
                             "SELF_ECHO_TAIL_S", "NOISE_FILTER_ENABLED",
                             "DEVICE_SPEECH_FILTER_ENABLED",
                             "MEDIA_VOICE_GATE_ENABLED",
                             "MEDIA_VOICE_GATE_REJECT_BELOW"]),
        ("Double-clap trigger", ["CLAP_TRIGGER_ENABLED", "CLAP_TRIGGER_ACTION",
                                 "CLAP_TRIGGER_WAKE", "CLAP_TRIGGER_COOLDOWN_S",
                                 "CLAP_TRIGGER_MIN_PEAK"]),
    ],
    "ai": [
        ("Brain", ["AI_BACKEND", "CLAUDE_MODEL", "LOCAL_LLM_MODEL",
                   "LOCAL_VISION_MODEL", "MODEL_ROUTING", "CLAUDE_OPTIONAL",
                   "LOCAL_LLM_FALLBACK", "LOCAL_VISION_FALLBACK",
                   "AMBIENT_LEARNING_FORCE_LOCAL", "GAME_MODE_ENABLED"]),
        ("Memory", ["LTM_ENABLED", "RAG_ENABLED"]),
        ("Speed", ["FAST_PATHS_ENABLED", "INSTANT_ACTIONS_MODE",
                   "PROMPT_FREEZE_QUIET_S",
                   "LOCAL_PREFIX_REPRIME", "LOCAL_BACKGROUND_MAX_DEFER_S",
                   "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S",
                   "BACKGROUND_TAG_STRICT", "LOCAL_REPRIME_AT_BOOT_S"]),
        ("Turn checker", ["TURN_CHECK_MODE", "TURN_CHECK_ESCALATE_MODEL"]),
        ("Background work", ["ENABLE_ORCHESTRATOR", "TEAMS_NUDGE_ENABLED"]),
        ("Spending caps", ["DAILY_BUDGET_USD", "DEEP_AUDIT_BUDGET_USD"]),
    ],
    "cameras": [
        ("Cameras", ["_view_cameras", "CAMERA_REOPEN_MAX_BACKOFF_S",
                     "USB_STORM_COOLDOWN_S", "CAMERA_STORM_PROBATION_S",
                     "CAMERA_CULPRIT_WINDOW_S", "CAMERA_CULPRIT_THRESHOLD",
                     "CAMERA_DIES_ON_OPEN_RETRY_S", "CAMERA_OPEN_MIN_GAP_S",
                     "TV_DETECT_ENABLED"]),
        ("Kinect", ["KINECT_ENABLED", "KINECT_AS_CAMERA",
                    "KINECT_PRESENCE_ENABLED", "KINECT_PRESENCE_STANDBY",
                    "KINECT_PRESENCE_WAKE", "KINECT_GREET_ON_ENTRY",
                    "KINECT_POSTURE_NUDGE", "KINECT_GAZE_ENABLED",
                    "KINECT_GESTURES_ENABLED", "KINECT_POINT_CONTROL_ENABLED",
                    "KINECT_GUARD_ENABLED",
                    "KINECT_SKELETON_OVERLAY_ENABLED"]),
        ("Kinect air-mouse", ["KINECT_AIR_MOUSE_ENABLED",
                              "AIR_MOUSE_REQUIRE_OPEN_PALM",
                              "AIR_MOUSE_ARM_RELAXES_GATE",
                              "AIR_MOUSE_FIST_RELEASES",
                              "AIR_MOUSE_PER_APP_DISABLE",
                              "KINECT_TWO_HAND_ENABLED"]),
        ("Experimental", ["AIR_CONTROL_ENABLED"]),
    ],
    "privacy": [
        ("Listening & watching", ["AMBIENT_LISTEN_ENABLED",
                                  "AMBIENT_SCREEN_ENABLED",
                                  "SCREENSHOT_PRIVACY_BLOCKLIST"]),
        ("Who can teach him", ["LEARN_ONLY_FROM_OWNER", "LEARN_FOLLOWUP_S",
                               "LEARN_VOICE_REJECT_BELOW"]),
        ("Faces", ["FACE_ID_ENABLED", "GREET_NEW_PEOPLE_ENABLED"]),
    ],
    "integrations": [
        ("Connections", ["_status_anthropic", "_status_porcupine",
                         "_status_azure_tts", "_status_elevenlabs",
                         "_status_bambu", "_status_govee", "_status_hue",
                         "_status_obs", "_status_deco", "_status_phone"]),
        ("Phone pings", ["PHONE_PING_ENABLED", "PHONE_PING_PRINT",
                         "PHONE_PING_CONFIRM", "PHONE_PING_SECURITY",
                         "PHONE_PING_ROBOT", "PHONE_PING_SUMMARY",
                         "PHONE_PING_SUMMARY_TIME", "PHONE_PING_MAX_PER_HOUR",
                         "PHONE_PING_QUIET_START", "PHONE_PING_QUIET_END",
                         "PHONE_PING_AWAY_MIN",
                         "PHONE_PING_CONFIRM_AFTER_MIN"]),
        ("Media", ["STREAMING_AUTO_FULLSCREEN"]),
        ("Notes (not read by JARVIS)", ["OBS_HOST_HINT", "OBS_PORT_HINT",
                                        "HUE_BRIDGE_IP_HINT"]),
    ],
    "advanced": [
        ("On screen", ["HUD_ENABLED", "HUD_MONITOR", "BRAIN_GLOW_ENABLED",
                       "BRAIN_GLOW_LABEL_S", "TRAY_ENABLED",
                       "RETICLE_OVERLAY_ENABLED"]),
        ("Behaviour", ["PUSHBACK_ENABLED", "MISSION_NARRATION_ENABLED",
                       "SCREEN_VISION_ENABLED", "PC_CONTROL_ENABLED",
                       "OVERNIGHT_UPGRADE_ENABLED"]),
        ("Debug", ["VAD_DEBUG"]),
        ("Web interface", ["WEB_INTERFACE_ENABLED", "WEB_INTERFACE_PORT",
                           "WEB_INTERFACE_BIND", "WEB_INTERFACE_TOKEN",
                           "DASHBOARD_SHOW_TRANSCRIPTS"]),
    ],
}


def tab_layout(tab: str) -> list[tuple[str, list[str]]]:
    """``[(section, [keys…]), …]`` for one tab: TAB_SECTIONS, plus an "Other"
    section for any row of that tab the layout doesn't list."""
    sections = [(name, [k for k in keys if SCHEMA.get(k, {}).get("tab") == tab])
                for name, keys in TAB_SECTIONS.get(tab, [])]
    listed = {k for _n, keys in sections for k in keys}
    other = [k for k, s in SCHEMA.items()
             if s.get("tab") == tab and k not in listed]
    if other:
        sections.append(("Other", other))
    return [(n, keys) for n, keys in sections if keys]


def row_section(key: str) -> str | None:
    """The sub-heading a row is shown under, or None for an unknown key."""
    tab = SCHEMA.get(key, {}).get("tab")
    for name, keys in tab_layout(tab) if tab else []:
        if key in keys:
            return name
    return None


def _copy_value(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def default_settings() -> dict:
    """The full template: every persisted key at its schema default.

    Mirrors core/config.py's current defaults (tools/user_settings.example.json
    is generated from this). Mutable defaults (lists/dicts) are copied so
    callers can't mutate the schema by editing the result."""
    return {key: _copy_value(SCHEMA[key]["default"]) for key in persisted_keys()}


def coerce_value(spec: dict, raw):
    """Coerce a raw value to the type its schema spec declares.

    Never raises on bad input — falls back to the spec default so a hand-edited
    settings file with a typo can't crash the GUI or a downstream reader. (The
    GUI's Save does NOT rely on this: it runs validate_value, which REFUSES a
    bad value and says why, instead of silently saving the default.)
    """
    spec = spec or {}
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
            # hard-off). null / "" / None -> the default (None); 3 / "3" / -1
            # -> int; a non-numeric value falls through to the default.
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
    except (TypeError, ValueError, OverflowError):
        return _copy_value(default)


def _fmt_num(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def validate_value(spec: dict, raw) -> tuple[object, str | None]:
    """STRICT parse of a value the owner entered: ``(value, None)`` when it is
    acceptable, ``(None, reason)`` when it must be refused.

    coerce_value is deliberately lenient (a hand-edited file must never crash a
    reader), which made the GUI's Save report "Saved." while writing the
    DEFAULT for "0,01", "20s", "nan", a negative VAD threshold or port 99999.
    This refuses those and says why, next to the field."""
    spec = spec or {}
    typ = spec.get("type")
    if typ in ("int", "float"):
        return _validate_number(spec, raw)
    if typ == "enum":
        val = str(raw)
        choices = [str(c) for c in (spec.get("choices") or [])]
        if val not in choices:
            return None, "choose one of: " + ", ".join(choices)
        return val, None
    if typ in ("str", "combo"):
        val = "" if raw is None else str(raw).strip()
        if spec.get("nonblank") and not val:
            return None, "can't be blank"
        return val, None
    if typ == "device":
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None, None
        if isinstance(raw, bool):
            return None, "pick a device"
        try:
            return int(raw), None
        except (TypeError, ValueError):
            return None, "pick a device"
    return coerce_value(spec, raw), None


def _validate_number(spec: dict, raw) -> tuple[object, str | None]:
    typ = spec.get("type")
    if isinstance(raw, bool):
        return None, "enter a number"
    if isinstance(raw, (int, float)):
        num = raw
    else:
        s = "" if raw is None else str(raw).strip()
        if not s:
            return None, "enter a number"
        if re.fullmatch(r"[+-]?\d+,\d+", s):
            return None, "use a dot for decimals (0.5, not 0,5)"
        try:
            num = float(s)
        except ValueError:
            return None, f"'{s}' is not a number"
    if isinstance(num, float) and (math.isnan(num) or math.isinf(num)):
        return None, "must be a real number"
    if typ == "int":
        if float(num) != int(num):
            return None, "must be a whole number"
        num = int(num)
    else:
        num = float(num)
    lo = spec.get("min", None if spec.get("allow_negative") else 0)
    hi = spec.get("max")
    if lo is not None:
        if spec.get("min_exclusive"):
            if num <= lo:
                return None, f"must be more than {_fmt_num(lo)}"
        elif num < lo:
            return None, f"must be at least {_fmt_num(lo)}"
    if hi is not None and num > hi:
        return None, f"must be at most {_fmt_num(hi)}"
    forbid = spec.get("forbid") or {}
    if num in forbid:
        return None, str(forbid[num])
    return num, None


def values_equal(a, b) -> bool:
    """Equality for settings values (1 == 1.0 counts; True != 1 does not)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


# ──────────────────────────────────────────────────────────────────────────
#  Reading and writing the settings file
# ──────────────────────────────────────────────────────────────────────────
class SettingsFileError(ValueError):
    """The settings file exists but is not a readable JSON object.

    Raised by read_settings_file — and therefore by every WRITER here, which
    reads the file first: nothing may be written over a document that could not
    be read, because that write would replace every key it failed to see."""


def _describe_read_error(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
    if isinstance(exc, UnicodeDecodeError):
        return "it is not UTF-8 text (saved as UTF-16?)"
    return f"{type(exc).__name__}: {exc}"


def read_settings_file(path: str | None = None) -> dict:
    """STRICT read of the raw document on disk.

    Missing or blank file → ``{}``. A JSON object → that dict, as written (no
    defaults layered in). Anything else → ``SettingsFileError`` naming the
    problem (line/column for a JSON error). Read as ``utf-8-sig``: PowerShell
    5.1's ``Set-Content/Out-File -Encoding utf8`` writes a BOM, which plain
    utf-8 rejected — and the old reader then treated the owner's whole file as
    empty."""
    if path is None:
        path = settings_path()
    if not os.path.exists(path):
        return {}
    name = os.path.basename(path) or path
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise SettingsFileError(
            f"{name} could not be read — {_describe_read_error(exc)}") from exc
    if not text.strip():
        return {}
    try:
        decoded = json.loads(text)
    except ValueError as exc:
        raise SettingsFileError(
            f"{name} is not valid JSON — {_describe_read_error(exc)}") from exc
    if not isinstance(decoded, dict):
        raise SettingsFileError(
            f"{name} must hold a JSON object {{ … }}, not a "
            f"{type(decoded).__name__}")
    return decoded


def settings_file_problem(path: str | None = None) -> str | None:
    """Why the settings file can't be read, or None when it is fine/absent."""
    try:
        read_settings_file(path)
        return None
    except SettingsFileError as exc:
        return str(exc)


def load_settings(path: str | None = None) -> dict:
    """Load settings, layering the on-disk file over the schema defaults.

    Missing file or missing keys fall back to defaults; every schema value is
    coerced to its type; unknown keys are passed through untouched. TOLERANT:
    an unreadable file loads as the defaults so a reader never crashes — but
    save_settings then REFUSES to write over it (see read_settings_file), so
    load-modify-save callers (the voice toggles) can no longer turn a typo
    into a wiped file. ``path`` defaults to ``settings_path()``."""
    merged = default_settings()
    try:
        raw = read_settings_file(path)
    except SettingsFileError:
        raw = {}
    # Coerce known keys; pass through unknown keys verbatim.
    for key, value in raw.items():
        spec = SCHEMA.get(key)
        merged[key] = coerce_value(spec, value) if spec else value
    return merged


def atomic_write_json(path: str, data: dict) -> None:
    """Write `data` as pretty JSON via temp-file + os.replace.

    Same crash-safe pattern as tray.py's `_send_command`: write to a temp file
    in the destination directory, then atomically rename over the target so a
    reader never observes a half-written file.
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


def _is_schema_default(spec: dict, value) -> bool:
    return values_equal(value, coerce_value(spec, spec.get("default")))


def save_settings(values: dict, path: str | None = None, *,
                  changed=None) -> None:
    """Persist a WHOLE settings document (the load → change one key → save
    pattern the voice toggles use), atomically.

    * Refuses (SettingsFileError) when the file on disk exists but can't be
      read — writing would delete every key the caller never saw.
    * A schema key that is ABSENT from the file and still at its schema
      default is not written: every save used to freeze all ~90 keys at that
      day's defaults, so a later release could never change a default the
      owner had never touched. A key already in the file, or a non-default
      value, is written as before.
    * Unknown keys in ``values`` pass through verbatim (LOCAL_VISION_MODEL
      used to rely on this; CAMERAS and calibration keys still do).
    * ``changed`` (2026-10-01): the keys the caller actually set. Only those
      are written (coerced, by the rules above); every other key stays
      EXACTLY as it is on disk. Without it every key is re-coerced, and
      load_settings coerces, so a hand-set value the schema can't express
      (an enum value newer than its choices, e.g. a Claude model id) loaded
      as the default and each voice toggle's load -> change one key -> save
      wrote that default back over it, silently. Only the caller knows
      "left as loaded" from "set to the default", so it must say: guessing
      from equality (the first fix) also dropped an explicit write of the
      default over a hand-typo and stopped healing malformed values.

    ``path`` defaults to ``settings_path()`` so a ``JARVIS_SETTINGS_PATH``
    redirect sends the write (and its temp file) to the throwaway file."""
    if path is None:
        path = settings_path()
    on_disk = read_settings_file(path)
    if changed is not None:
        changed = set(changed)
        out: dict = dict(on_disk)
        items = [(k, v) for k, v in values.items() if k in changed]
    else:
        out = {}
        items = list(values.items())
    for key, value in items:
        spec = SCHEMA.get(key)
        if spec is None or spec.get("type") not in _PERSISTED_TYPES:
            out[key] = value
            continue
        coerced = coerce_value(spec, value)
        if key not in on_disk and _is_schema_default(spec, coerced):
            continue
        out[key] = coerced
    atomic_write_json(path, out)


def update_settings(changes: dict, path: str | None = None) -> dict:
    """MERGE ``changes`` into the document as it is on disk right now and write
    it atomically; every other key is left exactly as found. Refuses
    (SettingsFileError) when the file can't be read. Returns the written doc."""
    if path is None:
        path = settings_path()
    out = dict(read_settings_file(path))
    for key, value in changes.items():
        spec = SCHEMA.get(key)
        if spec is not None and spec.get("type") in _PERSISTED_TYPES:
            out[key] = coerce_value(spec, value)
        else:
            out[key] = value
    atomic_write_json(path, out)
    return out


def save_changed_settings(changes: dict, path: str | None = None
                          ) -> tuple[dict, tuple]:
    """The Settings window's Save: write ONLY ``changes`` (the fields the owner
    edited) into the file as it is on disk at this moment.

    Nothing else is rewritten, so a setting a voice command changed while the
    window was open keeps the voice's value (the window's Save used to write
    EVERY field, reverting it), and untouched defaults stay out of the file.

    Applies the chat↔vision lockstep with the on-disk document as the "before"
    — unless the owner set LOCAL_VISION_MODEL in this same save, which wins.
    Returns ``(written_document, (vision_tag_or_None, reason))``. Refuses
    (SettingsFileError) when the file can't be read."""
    if path is None:
        path = settings_path()
    base = dict(read_settings_file(path))
    out = dict(base)
    for key, value in changes.items():
        spec = SCHEMA.get(key)
        if spec is not None and spec.get("type") in _PERSISTED_TYPES:
            out[key] = coerce_value(spec, value)
        else:
            out[key] = value
    lockstep: tuple = (None, "")
    if "LOCAL_LLM_MODEL" in changes and "LOCAL_VISION_MODEL" not in changes:
        lockstep = apply_vision_lockstep(base, out)
    atomic_write_json(path, out)
    return out, lockstep


def _current_settings_base(snapshot: dict, path: str | None = None) -> dict:
    """The CURRENT on-disk settings (merged over defaults), falling back to a
    copy of ``snapshot`` if the re-read fails. Kept for callers of the old
    whole-document Save; the window itself now writes only changed fields
    (save_changed_settings)."""
    try:
        return dict(load_settings(path))
    except Exception:
        return dict(snapshot)


def ensure_settings_file(path: str | None = None) -> dict:
    """Make sure a settings file exists (an EMPTY object when it didn't — the
    defaults live in core/config.py, and writing them all out would freeze
    them) and return the loaded settings. Used by "Open user_settings.json"."""
    if path is None:
        path = settings_path()
    if not os.path.exists(path):
        try:
            atomic_write_json(path, {})
        except OSError:
            pass
    return load_settings(path)


# ──────────────────────────────────────────────────────────────────────────
#  What is in effect
# ──────────────────────────────────────────────────────────────────────────
_MISSING = object()


def _config_value(key: str):
    """The value core/config.py has for ``key`` in THIS process (module default,
    env overrides and the settings file applied), or _MISSING. Lazy — importing
    this module never imports core."""
    mod = _model_lockstep()
    if mod is None:
        return _MISSING
    try:
        return mod.config_default(key, _MISSING)
    except Exception:
        return _MISSING


def effective_value(key: str, raw_doc: dict, config_lookup=None):
    """What JARVIS will use for ``key``: the file's value when the file has
    the key, else core/config.py's value (which may come from an env var —
    e.g. AUDIO_AUTOSWITCH_* — and differ from the schema default), else the
    schema default. Coerced to the row's type."""
    spec = SCHEMA[key]
    raw_doc = raw_doc if isinstance(raw_doc, dict) else {}
    if key in raw_doc:
        return coerce_value(spec, raw_doc[key])
    lookup = config_lookup or _config_value
    try:
        cfg = lookup(key)
    except Exception:
        cfg = _MISSING
    if cfg is not _MISSING and cfg is not None:
        if spec.get("type") == "text" and isinstance(cfg, tuple):
            cfg = list(cfg)
        return coerce_value(spec, cfg)
    return coerce_value(spec, _copy_value(spec.get("default")))


def effective_settings(raw_doc: dict, config_lookup=None) -> dict:
    """effective_value for every persisted key, plus the file's other keys."""
    out = {k: v for k, v in (raw_doc or {}).items() if k not in SCHEMA}
    for key in persisted_keys():
        out[key] = effective_value(key, raw_doc, config_lookup)
    return out


def cameras_summary(cameras) -> list[str]:
    """Human lines for the read-only CAMERAS view."""
    if not isinstance(cameras, (list, tuple)) or not cameras:
        return ["(no cameras configured)"]
    lines = []
    for i, cam in enumerate(cameras):
        if not isinstance(cam, dict):
            lines.append(f"• {cam!r}")
            continue
        label = str(cam.get("label") or cam.get("name") or f"camera {i + 1}")
        bits = []
        if cam.get("name"):
            bits.append(f"found by name '{cam.get('name')}'")
        if cam.get("type"):
            bits.append(f"type {cam.get('type')}")
        if cam.get("primary"):
            bits.append("primary")
        idx = cam.get("index", "?")
        lines.append(f"• {label} — index {idx}"
                     + (f" ({', '.join(bits)})" if bits else ""))
    return lines


def integration_status(spec: dict, find_spec=None) -> tuple[bool, str]:
    """Resolve a status row to (present, detail) WITHOUT exposing any secret.

    Only the PRESENCE of each env var (or config file) is checked — values are
    never read into the return. `match == "any"` means present if ANY listed
    env var is set; otherwise ALL must be set. A row whose env vars are all
    OPTIONAL (``optional_env``, e.g. OBS) is ready once its client package is
    installed, and says whether it runs on the defaults.
    """
    module = spec.get("requires_module")
    if module:
        fs = find_spec or importlib.util.find_spec
        try:
            have = fs(module) is not None
        except (ImportError, ValueError):
            have = False
        if not have:
            return (False, f"{spec.get('module_pip') or module} not installed")
    optional = spec.get("optional_env")
    if optional is not None:
        set_names = [e for e in optional if (os.environ.get(e) or "").strip()]
        if set_names:
            return (True, "ready — " + ", ".join(set_names) + " set")
        note = spec.get("defaults_note")
        return (True, f"ready — defaults ({note})" if note
                else "ready — defaults")
    envs = spec.get("secret_env") or []
    present_flags = [bool((os.environ.get(e) or "").strip()) for e in envs]
    cfg_present = False
    cfg_file = spec.get("config_file")
    if cfg_file:
        cfg_present = os.path.exists(os.path.join(DATA_DIR, cfg_file))

    if not envs and cfg_file:
        return (cfg_present, "configured" if cfg_present else "not configured")
    if not envs and not cfg_file:
        return (bool(module), "installed" if module else "not configured")

    if spec.get("match") == "any":
        present = any(present_flags) or cfg_present
    else:
        present = all(present_flags) or (cfg_present and not envs)
    return (present, "present" if present else "not set")


# Python packages each choice needs, probed with find_spec (never imported).
TTS_BACKEND_PACKAGES = {"edge": "edge_tts", "kokoro": "kokoro_onnx",
                        "pyttsx3": "pyttsx3", "xtts": "TTS"}
# RealtimeSTT imports pyaudio at import time, so without PyAudio the realtime
# session fails to build and JARVIS silently stays turn-based.
REALTIME_PACKAGES = ("RealtimeSTT", "RealtimeTTS", "pyaudio")
WAKE_ENGINE_PACKAGES = ("openwakeword", "pvporcupine")


def effective_warnings(values: dict, find_spec=None) -> dict[str, str]:
    """Rows whose saved value is NOT what JARVIS will actually do, with why.

    ``values`` maps setting → current value. Package presence is probed with
    importlib.util.find_spec only (nothing is imported). Never raises."""
    fs = find_spec or importlib.util.find_spec

    def have(mod: str) -> bool:
        try:
            return fs(mod) is not None
        except (ImportError, ValueError):
            return False
        except Exception:
            return False

    def truthy(v) -> bool:
        return coerce_value({"type": "bool", "default": False}, v)

    out: dict[str, str] = {}
    try:
        if str(values.get("VOICE_MODE", "")).strip().lower() == "realtime":
            missing = [m for m in REALTIME_PACKAGES if not have(m)]
            if missing:
                out["VOICE_MODE"] = (
                    "Not in effect: realtime needs " + ", ".join(missing)
                    + " — JARVIS falls back to turn_based.")
        if truthy(values.get("BARGE_IN_ENABLED")):
            if not any(have(m) for m in WAKE_ENGINE_PACKAGES):
                out["BARGE_IN_ENABLED"] = (
                    "Not in effect: no wake-word engine is installed "
                    "(openwakeword or pvporcupine).")
            else:
                out["BARGE_IN_ENABLED"] = (
                    "Only while the wake-word detector runs — it never starts "
                    "by itself; say 'start listening for the wake word' first.")
        if truthy(values.get("WAKE_WORD_AUTOSTART")) and not any(
                have(m) for m in WAKE_ENGINE_PACKAGES):
            out["WAKE_WORD_AUTOSTART"] = (
                "Not in effect: no wake-word engine is installed "
                "(openwakeword or pvporcupine).")
        backend = str(values.get("TTS_BACKEND", "")).strip().lower()
        pkg = TTS_BACKEND_PACKAGES.get(backend)
        if pkg and not have(pkg):
            out["TTS_BACKEND"] = (
                f"Not in effect: '{backend}' needs the {pkg} package, which "
                f"isn't installed — JARVIS falls back to another voice.")
        if truthy(values.get("VOICE_CLONE_ENABLED")) and not have("chatterbox"):
            out["VOICE_CLONE_ENABLED"] = (
                "Not in effect: chatterbox-tts isn't installed — the normal "
                "voice is used.")
        if truthy(values.get("AIR_CONTROL_ENABLED")) and truthy(
                values.get("KINECT_AIR_MOUSE_ENABLED")):
            out["AIR_CONTROL_ENABLED"] = (
                "Both hand engines are on — they fight over the cursor. Turn "
                "one off.")
    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────
#  Talking to the running JARVIS
# ──────────────────────────────────────────────────────────────────────────
def send_tray_command(cmd: str, path: str | None = None, **kwargs) -> bool:
    """Append ``{"cmd": cmd, "ts": …, "cid": …}`` to the tray command inbox
    with the same atomic temp+rename pattern as tray.py's ``_send_command`` (a
    copy of it — this process can't import tray.py). The running JARVIS drains
    the inbox every 0.5 s; "restart" runs its hardened teardown with no LLM
    involved. Returns True when the command was written. Never raises.

    The unique ``cid`` is the copy's half of the tray's v2.0.144 race fix: a
    drainer claim landing between our read and our replace makes us write the
    claimed commands back, and the drainer skips only a cid it has already
    seen — an entry without one ran twice (2026-10-01)."""
    target = path or TRAY_COMMANDS_FILE
    payload = {"cmd": cmd, "ts": time.time(), "cid": "s" + uuid.uuid4().hex}
    payload.update(kwargs)
    try:
        existing = []
        if os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw:
                    decoded, _ = json.JSONDecoder().raw_decode(raw)
                    if isinstance(decoded, list):
                        existing = decoded
            except Exception:
                existing = []
        existing.append(payload)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target) or ".",
                                   suffix=".tmp", prefix="settings_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(existing, f)
            os.replace(tmp, target)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise
        return True
    except Exception as exc:
        print(f"[settings] command write failed ({cmd}): {exc}",
              file=sys.stderr)
        return False


def fetch_pending_restart(values: dict, timeout: float = 1.5,
                          opener=None) -> set | None:
    """Ask the running JARVIS's web interface which saved settings it is NOT
    running with (GET /api/settings → rows flagged ``pending_restart``, i.e.
    the file value differs from the live core.config constant).

    Returns the set of setting names, or None when that can't be known (web
    interface off / unreachable / refused). Loopback only, short timeout; the
    GUI calls it on a background thread. Never raises."""
    try:
        if not coerce_value({"type": "bool", "default": False},
                            values.get("WEB_INTERFACE_ENABLED")):
            return None
        port = int(values.get("WEB_INTERFACE_PORT") or 8766)
        bind = str(values.get("WEB_INTERFACE_BIND") or "").strip()
        host = "127.0.0.1" if bind in ("", "0.0.0.0", "::", "localhost",
                                       "127.0.0.1") else bind
        import urllib.request
        req = urllib.request.Request(f"http://{host}:{port}/api/settings")
        token = str(values.get("WEB_INTERFACE_TOKEN") or "").strip()
        if token:
            req.add_header("X-Auth-Token", token)
        open_ = opener or urllib.request.urlopen
        with open_(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("settings") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return None
        return {str(r.get("name")) for r in rows
                if isinstance(r, dict) and r.get("pending_restart")}
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
#  One window at a time
# ──────────────────────────────────────────────────────────────────────────
# Every tray click used to open ANOTHER always-on-top window. Now the first
# window holds a named mutex; a second launch asks it to come to the front (a
# small request file it polls) and exits.
_ERROR_ALREADY_EXISTS = 183


def instance_tag(path: str | None = None) -> str:
    """A short id for the settings file this window edits (windows editing
    different files — prod vs staging — don't block each other)."""
    p = os.path.normcase(os.path.abspath(path or settings_path()))
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:12]


def acquire_single_instance(tag: str):
    """Take the per-file named mutex. Returns a handle to keep for the
    process's lifetime, or None when another settings window already holds it.
    Off Windows (or on any error) returns a truthy placeholder — one-window is
    best-effort, never a reason not to open."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_wchar_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k32.CreateMutexW(None, False,
                                  f"Local\\JARVIS-SettingsWindow-{tag}")
        err = ctypes.get_last_error()
        if not handle:
            return True
        if err == _ERROR_ALREADY_EXISTS:
            k32.CloseHandle(handle)
            return None
        return handle
    except Exception:
        return True


def release_single_instance(handle) -> None:
    if sys.platform != "win32" or handle in (None, True):
        return
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.CloseHandle(handle)
    except Exception:
        pass


def _focus_file(tag: str, directory: str | None = None) -> str:
    return os.path.join(directory or tempfile.gettempdir(),
                        f"jarvis_settings_{tag}.focus")


def _pid_file(tag: str, directory: str | None = None) -> str:
    return os.path.join(directory or tempfile.gettempdir(),
                        f"jarvis_settings_{tag}.pid")


def request_focus(tag: str, directory: str | None = None) -> bool:
    """Second launch: ask the open window to come to the front. Also lets that
    window take the foreground (Windows only grants it to the process the user
    just clicked). Never raises."""
    ok = False
    try:
        with open(_focus_file(tag, directory), "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        ok = True
    except OSError:
        ok = False
    if sys.platform == "win32":
        try:
            with open(_pid_file(tag, directory), "r", encoding="utf-8") as f:
                pid = int(f.read().strip() or "0")
            if pid > 0:
                import ctypes
                ctypes.windll.user32.AllowSetForegroundWindow(pid)
        except Exception:
            pass
    return ok


def consume_focus_request(tag: str, directory: str | None = None) -> bool:
    """True (once) when a second launch asked this window to come forward."""
    p = _focus_file(tag, directory)
    try:
        if os.path.exists(p):
            os.remove(p)
            return True
    except OSError:
        pass
    return False


def write_pid_file(tag: str, directory: str | None = None) -> None:
    try:
        with open(_pid_file(tag, directory), "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────────
#  VRAM budget bridge  (import-safe: the engine is stdlib-only and never
#  raises; the heavy work lives in core/vram_budget.py)
# ──────────────────────────────────────────────────────────────────────────
# Keys whose LIVE value feeds the VRAM budget — changing any of these re-runs
# the prediction in the GUI. ``MODEL_ROUTING::vision`` is the flattened form
# of the routing row's vision dropdown.
VRAM_WATCH_KEYS = (
    "LOCAL_LLM_MODEL",
    "LOCAL_VISION_MODEL",
    "MODEL_ROUTING::vision",
    "LOCAL_VISION_FALLBACK",
    "SCREEN_VISION_ENABLED",
    "RAG_ENABLED",
    "KINECT_ENABLED",
    # 2026-09-30: Whisper on cuda:1 lives on the SECOND card; the panel used to
    # charge its 1.5 GB to the 3090 regardless.
    "WHISPER_DEVICE",
)


# One-shot guard so a missing core.model_lockstep is REPORTED once instead of
# degrading silently on every keystroke (the live budget callback is hot).
_LOCKSTEP_IMPORT_WARNED = [False]

# Mirrors core.model_lockstep.LOCKSTEP_TEXT_ONLY — the one reason the Save
# button explains out loud ("your new brain can't see, so vision stayed put").
# Duplicated as a literal only because this module must not import core at
# import time; tests/test_model_lockstep.py pins the two together.
LOCKSTEP_TEXT_ONLY_REASON = "text-only"


def _model_lockstep():
    """Import core.model_lockstep lazily, returning the module or None.

    Lazy for the same reason core.vram_budget is: this file's contract is that
    importing it costs nothing but stdlib, so a bare CI runner / the tray's
    schema use never pulls core in. A failure is PRINTED once (stderr) rather
    than swallowed — the two things this module needs it for (the VRAM budget's
    config fallback and the vision lockstep) both degrade to a wrong-but-quiet
    answer otherwise, which is exactly the failure mode the audit found."""
    try:
        from core import model_lockstep  # lazy: keeps module import stdlib-only
        return model_lockstep
    except Exception as exc:             # pragma: no cover - in-repo module
        if not _LOCKSTEP_IMPORT_WARNED[0]:
            _LOCKSTEP_IMPORT_WARNED[0] = True
            print(f"[settings] core.model_lockstep unavailable ({exc}) — the "
                  f"VRAM budget will ignore config defaults and Save will not "
                  f"keep the vision model in lockstep.", file=sys.stderr)
        return None


def _config_default(key: str):
    """The value core/config.py ships for ``key``, or None.

    A key the settings document omits is still LIVE at its config value (core.
    config only overrides keys the file actually contains), so anything
    reasoning about the EFFECTIVE settings has to consult config for the keys
    the document omits — see resolve_vram_values()."""
    mod = _model_lockstep()
    return None if mod is None else mod.config_default(key)


def apply_vision_lockstep(base: dict, out: dict) -> tuple[str | None, str]:
    """Keep LOCAL_VISION_MODEL in lockstep when a Save repoints LOCAL_LLM_MODEL.

    ``base`` is the document as it stands ON DISK (it still holds the OLD chat
    tag); ``out`` is the document about to be written (it holds the NEW one).
    Mutates ``out`` in place and returns ``(new_vision_tag_or_None, reason)``
    from the shared rule — see core.model_lockstep.vision_lockstep_decision.

    WHY THE GUI NEEDS THIS: the chat-model combo is a real model-switch entry
    point, but until 2026-08-20 only the VOICE path applied the lockstep. A
    Save that moved LOCAL_LLM_MODEL left LOCAL_VISION_MODEL pointing at the
    old tag — forking the one-multimodal-brain config into a genuine second
    VLM co-load, and permanently: the voice path then reads the mismatch as a
    user-pinned VLM and refuses to repair it. Never raises."""
    mod = _model_lockstep()
    if mod is None:
        return (None, "unavailable")
    try:
        old_chat = base.get("LOCAL_LLM_MODEL") or mod.config_default(
            "LOCAL_LLM_MODEL")
        new_chat = out.get("LOCAL_LLM_MODEL")
        cur_vision = (out.get("LOCAL_VISION_MODEL")
                      or base.get("LOCAL_VISION_MODEL")
                      or mod.config_default("LOCAL_VISION_MODEL"))
        tag, reason = mod.vision_lockstep_decision(old_chat, new_chat,
                                                   cur_vision)
        if tag:
            out["LOCAL_VISION_MODEL"] = tag
        return (tag, reason)
    except Exception as exc:             # pragma: no cover - defensive
        print(f"[settings] vision lockstep skipped: {exc}", file=sys.stderr)
        return (None, "error")


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


def resolve_vram_values(widget_values: dict, settings: dict) -> dict:
    """Resolve every VRAM_WATCH_KEYS entry to its EFFECTIVE value.

    ``widget_values`` is what the live widgets currently hold (only the keys
    whose widget exists); ``settings`` is the loaded settings document.
    Resolution order is widget → saved settings → core/config.py constant, so
    an unsaved edit beats a saved value which beats the shipped default.

    The LAST fallback is load-bearing: a key the document omits is still live
    at its core.config value. Without it the engine once saw LOCAL_VISION_MODEL
    as ABSENT, took its legacy branch and charged a phantom flat 7.3 GB VLM on
    the SHIPPED default config (2026-08-20 audit).

    Pure and Tk-free so the tests drive the same resolution the GUI does."""
    out: dict = {}
    settings = settings if isinstance(settings, dict) else {}
    for key in VRAM_WATCH_KEYS:
        if key in widget_values:
            out[key] = widget_values[key]
            continue
        if "::" in key:                     # flattened routing sub-key
            root_key, fn = key.split("::", 1)
            cur = settings.get(root_key)
            if not isinstance(cur, dict):
                cur = _config_default(root_key)
            if isinstance(cur, dict) and fn in cur:
                out[key] = cur.get(fn)
            continue
        if key in settings:
            out[key] = settings.get(key)
            continue
        val = _config_default(key)
        if val is not None:
            out[key] = val
    return out


def budget_from_live_values(values: dict, total_mb=None) -> dict | None:
    """Run the VRAM prediction from a flat dict of CURRENT widget values.

    This is the value→budget function the live GUI callback calls (and the one
    the tests exercise — no Tk/pixels involved). Returns the predict_budget()
    dict, or None when the engine is unavailable. Never raises."""
    vb = _load_vram_budget()
    if vb is None:
        return None
    settings = dict(values) if isinstance(values, dict) else {}
    try:
        return vb.predict_budget(settings, total_mb=total_mb)
    except Exception:
        return None


def budget_parts_text(budget: dict) -> str:
    """The per-component breakdown line under the VRAM bar, e.g.
    "26B 16 · vision (shared with chat) 0 · Whisper (on cuda:1) · …"."""
    parts = []
    for c in (budget or {}).get("components", []):
        if c.get("elsewhere"):
            parts.append(str(c.get("label")))
            continue
        gb = c.get("mb", 0) / 1024.0
        gb_s = f"{int(round(gb))}" if abs(gb - round(gb)) < 0.05 else f"{gb:.1f}"
        tag = " (on-demand)" if c.get("ondemand") else ""
        parts.append(f"{c.get('label')} {gb_s}{tag}")
    return " · ".join(parts)


def vram_bar_layout(budget: dict, width) -> dict:
    """Where the VRAM bar's pieces go, ``width`` pixels wide, for a
    predict_budget() result.

    Within budget the bar spans the usable budget and the fill is the predicted
    peak. Over budget the fill used to clamp at full width, so 101% and 137%
    looked the same (GUI_REVIEW B16). The bar then spans the predicted PEAK
    instead: the solid fill stops at ``limit_px``, where the budget runs out,
    and the rest of the bar is the overage, drawn hatched and labelled with
    ``over_label``. ``limit_px`` is None within budget; ``over`` says which
    case was drawn. Never raises.

    2026-10-02: a card at or under the headroom has a usable budget of 0, and
    predict_budget() calls ANY load on it over (``total > budget``) — but this
    returned an empty, unmarked bar for it, so the one card with no room at
    all looked like the card with nothing loaded. The whole bar is the
    overage there: the limit sits at 0 and the label carries the full load."""
    try:
        width = max(1, int(width))
        total = max(0, int((budget or {}).get("total_mb") or 0))
        cap = max(0, int((budget or {}).get("budget_mb") or 0))
    except (TypeError, ValueError):
        return {"fill_px": 0, "limit_px": None, "over_label": "",
                "over": False}
    if total <= cap:
        frac = min(1.0, total / cap) if cap > 0 else 0.0
        return {"fill_px": int(width * frac), "limit_px": None,
                "over_label": "", "over": False}
    limit_px = int(width * cap / total)
    return {"fill_px": limit_px, "limit_px": limit_px,
            "over_label": f"+{(total - cap) / 1024.0:.1f} GB over",
            "over": True}


# ──────────────────────────────────────────────────────────────────────────
#  Small Tk-free helpers the GUI uses (tested directly)
# ──────────────────────────────────────────────────────────────────────────
WHEEL_NOTCH = 120


def wheel_steps(accum: float, delta) -> tuple[int, float]:
    """A mouse-wheel ``delta`` → whole scroll steps, carrying the remainder.

    A mouse notch is 120 per step; a precision touchpad sends many small
    deltas (8, 15, 30…), which ``int(delta / 120)`` rounded to 0 — the page
    never scrolled. Returns ``(steps, new_accum)``; steps < 0 scrolls up."""
    try:
        accum = float(accum) + float(delta)
    except (TypeError, ValueError):
        return 0, float(accum or 0.0)
    whole = int(accum / WHEEL_NOTCH)          # toward zero
    accum -= whole * WHEEL_NOTCH
    return -whole, accum


def apply_dark_theme(style, root) -> None:
    """Dark ttk styling, INCLUDING the read-only state and the dropdown lists.

    clam draws a read-only combobox's field with its frame grey (#dcdad5); with
    our light-grey text on top the chosen value was nearly invisible, and the
    popdown list stayed white. ``style``/``root`` are a ttk.Style and the Tk
    root (duck-typed so the tests can check the calls without a display)."""
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=BG, foreground=FG, font=FONT)
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=FIELD_BG, foreground=FG,
                    padding=(12, 6), font=FONT)
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG, font=FONT)
    style.configure("Help.TLabel", background=BG, foreground=MUTED,
                    font=FONT_SMALL)
    style.configure("Head.TLabel", background=BG, foreground=FG, font=FONT_BOLD)
    style.configure("Section.TLabel", background=BG, foreground=ACCENT,
                    font=FONT_SECTION)
    style.configure("TCheckbutton", background=BG, foreground=FG, font=FONT,
                    indicatorbackground=FIELD_BG, indicatorforeground=FG)
    style.map("TCheckbutton", background=[("active", BG)],
              indicatorbackground=[("selected", ACCENT), ("active", FIELD_BG)])
    style.configure("TCombobox", fieldbackground=FIELD_BG, background=FIELD_BG,
                    foreground=FG, arrowcolor=FG, bordercolor=BORDER,
                    lightcolor=FIELD_BG, darkcolor=FIELD_BG,
                    selectbackground=ACCENT, selectforeground="#ffffff",
                    insertcolor=FG, font=FONT)
    # The fix: every STATE the field can be drawn in, not just the default.
    style.map("TCombobox",
              fieldbackground=[("readonly", FIELD_BG), ("disabled", BG),
                               ("focus", FIELD_BG)],
              foreground=[("readonly", FG), ("disabled", MUTED)],
              background=[("readonly", FIELD_BG), ("active", FIELD_BG),
                          ("pressed", FIELD_BG)],
              selectbackground=[("readonly", FIELD_BG)],
              selectforeground=[("readonly", FG)],
              arrowcolor=[("disabled", MUTED), ("active", "#ffffff")])
    style.configure("TButton", background=FIELD_BG, foreground=FG, font=FONT,
                    padding=(10, 5), bordercolor=BORDER)
    style.map("TButton", background=[("active", ACCENT), ("disabled", BG)],
              foreground=[("disabled", MUTED)])
    style.configure("Vertical.TScrollbar", background=FIELD_BG,
                    troughcolor=BG, arrowcolor=FG, bordercolor=BORDER)
    # The popdown LIST is a plain Tk listbox: styled via the option database.
    for opt, val in (("background", FIELD_BG), ("foreground", FG),
                     ("selectBackground", ACCENT),
                     ("selectForeground", "#ffffff"), ("font", FONT)):
        root.option_add(f"*TCombobox*Listbox.{opt}", val)


def make_combo_wheel_guard(scroll_page):
    """A <MouseWheel> handler for a combobox: the wheel changes the value ONLY
    when the combobox has keyboard focus; otherwise it scrolls the page, as the
    owner meant. (ttk binds the wheel on every combobox, so scrolling down a
    tab silently changed whatever dropdown passed under the pointer.)"""
    def _guard(event):
        w = event.widget
        try:
            focused = str(w.focus_get()) == str(w)
        except Exception:
            focused = False
        if not focused:
            scroll_page(event)
            return "break"
        try:
            values = list(w.cget("values") or ())
            if values:
                step = -1 if getattr(event, "delta", 0) > 0 else 1
                cur = w.current()
                cur = 0 if cur is None or cur < 0 else cur
                new = max(0, min(len(values) - 1, cur + step))
                if new != cur:
                    w.current(new)
                    try:
                        w.event_generate("<<ComboboxSelected>>")
                    except Exception:
                        pass
        except Exception:
            pass
        return "break"
    return _guard


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
    parser.add_argument(
        "--selftest", action="store_true",
        help="Check that the core modules this window needs import from here "
             "(the way the tray launches it), print JSON, and exit — no window.",
    )
    return parser.parse_args(argv)


def resolve_start_tab(tab) -> int:
    """Map a `--tab` name to its index in TAB_ORDER (0 when unset/unknown)."""
    if tab in TAB_ORDER:
        return TAB_ORDER.index(tab)
    return 0


def selftest() -> dict:
    """The --selftest report: can this process (started the way the tray
    starts it) import what the window needs? No window, no file writes."""
    report: dict = {
        "project_dir_on_path": any(
            os.path.normcase(os.path.abspath(p or os.curdir))
            == os.path.normcase(PROJECT_DIR) for p in sys.path),
        "imports": {},
    }
    for name in ("core.model_lockstep", "core.vram_budget"):
        try:
            importlib.import_module(name)
            report["imports"][name] = "ok"
        except Exception as exc:
            report["imports"][name] = f"{type(exc).__name__}: {exc}"
    report["vram_panel"] = _load_vram_budget() is not None
    report["vision_lockstep"] = _model_lockstep() is not None
    report["tkinter"] = importlib.util.find_spec("tkinter") is not None
    report["ok"] = (all(v == "ok" for v in report["imports"].values())
                    and report["vram_panel"] and report["vision_lockstep"])
    return report


def set_dpi_awareness() -> None:
    """System-DPI-aware so Windows doesn't bitmap-stretch (blur) the window on
    a scaled display. Tk then sizes its point-sized fonts for the real DPI; the
    window's pixel sizes are scaled in run_gui. Never raises."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # SYSTEM_DPI_AWARE
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
#  ── GUI ──   (everything below requires tkinter; kept out of import-time)
# ──────────────────────────────────────────────────────────────────────────
class _Field:
    """One editable row: how to read its widget, the widget value it opened
    with (a field whose widget still holds that is UNCHANGED and never saved),
    and the labels that show its error / warning / restart badge."""

    def __init__(self, key: str, spec: dict, get_raw, set_raw):
        self.key = key
        self.spec = spec
        self.get_raw = get_raw
        self.set_raw = set_raw
        self.initial_raw = None
        self.error_label = None
        self.note_label = None
        self.badge_label = None
        self.combo = None


class SettingsApp:
    """The Settings window. ``tk``/``ttk``/``messagebox`` are the tkinter
    modules (injected so the tests can drive the whole window with fakes and no
    display); every other dependency with side effects — the settings file,
    config lookups, the device list, the Ollama and web probes, the tray inbox,
    opening a file — is injectable too."""

    def __init__(self, tk, ttk, messagebox, *, start_tab: int = 0,
                 path: str | None = None, config_lookup=None,
                 audio_devices=None, model_probe=None, running_probe=None,
                 find_spec=None, command_path: str | None = None,
                 file_opener=None, focus_tag: str | None = None,
                 focus_dir: str | None = None, poll_ms: int = 250,
                 total_vram_mb=None):
        self.tk, self.ttk, self.mb = tk, ttk, messagebox
        self.path = path or settings_path()
        self.config_lookup = config_lookup
        self._audio_devices = audio_devices      # (devices, hostapis) or None
        self._model_probe = model_probe or installed_ollama_models
        self._running_probe = running_probe or fetch_pending_restart
        self._find_spec = find_spec
        self._command_path = command_path
        self._file_opener = file_opener
        self._focus_tag = focus_tag
        self._focus_dir = focus_dir
        self._poll_ms = poll_ms
        self.file_error = settings_file_problem(self.path)
        raw = {}
        if not self.file_error:
            try:
                raw = read_settings_file(self.path)
            except SettingsFileError as exc:
                self.file_error = str(exc)
        self.raw = raw
        self.values = effective_settings(raw, config_lookup)
        self.snapshot = {k: _copy_value(self.values[k]) for k in persisted_keys()}
        self.fields: dict[str, _Field] = {}
        self.device_pickers: dict[str, dict] = {}
        self.help_labels: list = []
        self.saved_this_session: set = set()
        self.pending_running: set | None = None
        self._async: dict = {}
        self._wheel_accum = {"v": 0.0}
        self.vram = _load_vram_budget()
        self.vram_widgets: dict = {}
        self.vram_total_mb = total_vram_mb
        if self.vram is not None and self.vram_total_mb is None:
            try:
                self.vram_total_mb = self.vram.total_vram_mb()
            except Exception:
                self.vram_total_mb = None
        self.closed = False
        self._build(start_tab)

    # ── construction ──────────────────────────────────────────────────
    def _build(self, start_tab: int) -> None:
        tk, ttk = self.tk, self.ttk
        root = tk.Tk()
        self.root = root
        title = "JARVIS Settings"
        if os.path.normcase(os.path.abspath(self.path)) != os.path.normcase(
                os.path.abspath(SETTINGS_PATH)):
            title += f" — {self.path}"
        root.title(title)
        root.configure(bg=BG)
        scale = 1.0
        try:
            scale = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        except Exception:
            scale = 1.0
        self.scale = scale
        root.geometry(f"{int(780 * scale)}x{int(720 * scale)}")
        root.minsize(int(560 * scale), int(460 * scale))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-s>", lambda e: (self.save(), "break")[1])
        root.bind("<Control-S>", lambda e: (self.save(), "break")[1])
        root.bind("<Escape>", lambda e: (self.close(), "break")[1])

        style = ttk.Style()
        apply_dark_theme(style, root)

        # Banner: an unreadable settings file blocks saving, loudly. Packed
        # only while there is something to say.
        self.notebook = ttk.Notebook(root)
        self.banner = tk.Label(root, text="", bg=BG, fg=ERROR, font=FONT,
                               anchor="w", justify="left",
                               wraplength=int(740 * scale))
        self._banner_packed = False
        self._refresh_banner()
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(6, 4))
        self._notebook_packed = True
        self._tab_scrollers: list = []
        for tab_key in TAB_ORDER:
            self._build_tab(tab_key)
        root.bind_all("<MouseWheel>", self._on_wheel)

        # Initial per-row notes (effective state) and the VRAM bar.
        self._refresh_warnings()
        if self.vram_widgets:
            try:
                root.after(0, self.update_budget)
            except Exception:
                self.update_budget()
        try:
            self.notebook.select(start_tab)
        except Exception:
            pass

        # ── bottom bar ──
        bar = ttk.Frame(root, style="TFrame")
        bar.pack(fill="x", padx=10, pady=(0, 10))
        self.status_var = tk.StringVar(value="")
        note = ttk.Label(bar, text=RESTART_NOTE, style="Help.TLabel")
        note.pack(side="top", anchor="w")
        tk.Label(bar, textvariable=self.status_var, bg=BG, fg=FG,
                 font=FONT_SMALL, anchor="w", justify="left",
                 wraplength=int(740 * scale)).pack(side="top", fill="x")
        buttons = ttk.Frame(bar, style="TFrame")
        buttons.pack(side="top", fill="x", pady=(4, 0))
        self.buttons = {}
        self.buttons["close"] = ttk.Button(buttons, text="Close",
                                           command=self.close)
        self.buttons["close"].pack(side="right", padx=(6, 0))
        self.buttons["restart"] = ttk.Button(
            buttons, text="Save & restart JARVIS",
            command=lambda: self.save(restart=True))
        self.buttons["restart"].pack(side="right", padx=(6, 0))
        self.buttons["save"] = ttk.Button(buttons, text="Save",
                                          command=self.save)
        self.buttons["save"].pack(side="right")
        self.buttons["open"] = ttk.Button(buttons,
                                          text="Open user_settings.json",
                                          command=self.open_json)
        self.buttons["open"].pack(side="right", padx=(0, 6))
        self._sync_save_buttons()

        # Background work, polled from the Tk thread (Tk isn't thread-safe).
        self._start_async("models", lambda: (
            self._model_probe(),
            self._model_probe(include_vision=True)
            if self._probe_takes_vision() else None))
        self._start_async("running",
                          lambda: self._running_probe(dict(self.values)))
        self._tick()
        self.raise_window()

    def _probe_takes_vision(self) -> bool:
        try:
            import inspect
            return "include_vision" in inspect.signature(
                self._model_probe).parameters
        except Exception:
            return False

    def _refresh_banner(self) -> None:
        if not self.file_error:
            self.banner.configure(text="")
            return
        self.banner.configure(
            text=("⚠ Can't read the settings file, so Save is disabled "
                  f"(saving would erase what it can't read): "
                  f"{self.file_error}. Fix it with 'Open user_settings.json' "
                  f"and reopen this window. File: {self.path}"))
        if not self._banner_packed:
            self._banner_packed = True
            try:
                if getattr(self, "_notebook_packed", False):
                    self.banner.pack(fill="x", padx=10, pady=(8, 0),
                                     before=self.notebook)
                else:
                    self.banner.pack(fill="x", padx=10, pady=(8, 0))
            except Exception:
                pass

    def _sync_save_buttons(self) -> None:
        state = "disabled" if self.file_error else "normal"
        for name in ("save", "restart"):
            b = getattr(self, "buttons", {}).get(name)
            if b is not None:
                try:
                    b.configure(state=state)
                except Exception:
                    pass

    def _scrollable(self, tab_key):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.notebook, style="TFrame")
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical",
                                  command=canvas.yview)
        inner = ttk.Frame(canvas, style="TFrame")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        tab_help: list = []

        def _on_canvas_resize(event):
            # Stretch the content to the window width, and re-wrap the help.
            try:
                canvas.itemconfigure(win, width=event.width)
                wrap = max(280, int(event.width) - 24)
                for lbl in tab_help:
                    lbl.configure(wraplength=wrap)
            except Exception:
                pass
        canvas.bind("<Configure>", _on_canvas_resize)

        def _scroll_page(event):
            steps, self._wheel_accum["v"] = wheel_steps(
                self._wheel_accum["v"], getattr(event, "delta", 0))
            if steps:
                canvas.yview_scroll(steps, "units")

        # The page wheel is ONE application-wide binding that scrolls the
        # SELECTED tab (see _on_wheel) — B17 was every tab scrolling the last
        # one built, and the Enter/Leave rebinding that fixed it dropped the
        # wheel whenever the pointer sat on a label (Tk sends the parent a
        # <Leave> when the pointer moves onto a child).
        self._tab_scrollers.append(_scroll_page)
        inner.columnconfigure(1, weight=1)
        return outer, inner, tab_help, _scroll_page

    def _on_wheel(self, event):
        try:
            idx = int(self.notebook.index("current"))
        except Exception:
            idx = 0
        if 0 <= idx < len(self._tab_scrollers):
            self._tab_scrollers[idx](event)

    def _help(self, parent, text, row, tab_help, color=MUTED):
        lbl = self.tk.Label(parent, text=text, bg=BG, fg=color,
                            font=FONT_SMALL, anchor="w", justify="left",
                            wraplength=int(700 * self.scale))
        lbl.grid(row=row, column=0, columnspan=3, sticky="we", padx=(2, 0),
                 pady=(0, 2))
        tab_help.append(lbl)
        return lbl

    def _row_labels(self, parent, field, row, tab_help) -> int:
        """Error / effective-state / restart labels + help under a control."""
        spec = field.spec
        field.error_label = self.tk.Label(parent, text="", bg=BG, fg=ERROR,
                                          font=FONT_SMALL, anchor="w",
                                          justify="left")
        field.error_label.grid(row=row, column=0, columnspan=3, sticky="we",
                               padx=(2, 0))
        field.error_label.grid_remove()
        row += 1
        field.note_label = self.tk.Label(parent, text="", bg=BG, fg=WARN,
                                         font=FONT_SMALL, anchor="w",
                                         justify="left",
                                         wraplength=int(700 * self.scale))
        field.note_label.grid(row=row, column=0, columnspan=3, sticky="we",
                              padx=(2, 0))
        field.note_label.grid_remove()
        tab_help.append(field.note_label)
        row += 1
        if spec.get("help"):
            self._help(parent, spec["help"], row, tab_help)
        row += 1
        return row

    def _badge(self, parent, row):
        lbl = self.tk.Label(parent, text="", bg=BG, fg=WARN, font=FONT_SMALL,
                            anchor="e")
        lbl.grid(row=row, column=2, sticky="e", padx=(6, 2))
        return lbl

    def _combo(self, parent, var, values, readonly, scroll_page):
        combo = self.ttk.Combobox(parent, textvariable=var, values=values,
                                  state="readonly" if readonly else "normal",
                                  width=34)
        combo.bind("<MouseWheel>", make_combo_wheel_guard(scroll_page))
        return combo

    def _build_tab(self, tab_key: str) -> None:
        tk, ttk = self.tk, self.ttk
        outer, inner, tab_help, scroll_page = self._scrollable(tab_key)
        self.help_labels.append(tab_help)
        row = 0
        if tab_key == "ai":
            row = self._build_vram_panel(inner, row, tab_help)
        ordered = [(sec, key) for sec, keys in tab_layout(tab_key)
                   for key in keys]
        section = None
        for sec, key in ordered:
            spec = SCHEMA[key]
            if sec != section:
                section = sec
                ttk.Label(inner, text=sec, style="Section.TLabel").grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=2,
                    pady=(12 if row else 2, 2))
                row += 1
            typ = spec.get("type")
            label = spec.get("label", key)

            if typ == "status":
                present, detail = integration_status(spec,
                                                     find_spec=self._find_spec)
                dot = "●" if present else "○"
                color = OK_GREEN if present else MUTED
                tk.Label(inner, text=f"{dot} {label}: {detail}", bg=BG,
                         fg=color, font=FONT, anchor="w").grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=2,
                    pady=(2, 0))
                row += 1
                if spec.get("help"):
                    self._help(inner, spec["help"], row, tab_help)
                row += 1
                continue

            if typ == "view":
                src = spec.get("source_key")
                val = self.raw.get(src, _MISSING) if isinstance(
                    self.raw, dict) else _MISSING
                if val is _MISSING:
                    try:
                        val = (self.config_lookup or _config_value)(src)
                    except Exception:
                        val = _MISSING
                lines = cameras_summary(None if val is _MISSING else val)
                ttk.Label(inner, text=label, style="TLabel").grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=2,
                    pady=(4, 0))
                row += 1
                tk.Label(inner, text="\n".join(lines), bg=FIELD_BG, fg=FG,
                         font=FONT_SMALL, anchor="w", justify="left").grid(
                    row=row, column=0, columnspan=3, sticky="we", padx=2,
                    pady=(0, 2))
                row += 1
                if spec.get("help"):
                    self._help(inner, spec["help"], row, tab_help)
                row += 1
                continue

            value = self.values.get(key)

            if typ == "bool":
                var = tk.BooleanVar(value=bool(value))
                f = _Field(key, spec, var.get, var.set)
                ttk.Checkbutton(inner, text=label, variable=var).grid(
                    row=row, column=0, columnspan=2, sticky="w", padx=2,
                    pady=(4, 0))
                f.badge_label = self._badge(inner, row)
                self._register(f, var)
                row += 1
                row = self._row_labels(inner, f, row, tab_help)
                continue

            if typ == "text":
                ttk.Label(inner, text=label).grid(
                    row=row, column=0, columnspan=2, sticky="w", padx=2,
                    pady=(6, 2))
                badge = self._badge(inner, row)
                row += 1
                txt = tk.Text(inner, height=4, width=48, bg=FIELD_BG, fg=FG,
                              insertbackground=FG, font=FONT, relief="flat",
                              padx=6, pady=4)
                cur = value if isinstance(value, list) else []
                txt.insert("1.0", "\n".join(str(x) for x in cur))
                txt.grid(row=row, column=0, columnspan=3, sticky="we",
                         padx=2, pady=(0, 2))

                def _get(t=txt):
                    return t.get("1.0", "end-1c")

                def _set(v, t=txt):
                    t.delete("1.0", "end")
                    t.insert("1.0", "\n".join(v) if isinstance(v, list)
                             else str(v))
                f = _Field(key, spec, _get, _set)
                f.badge_label = badge
                self._register(f, None)
                row += 1
                row = self._row_labels(inner, f, row, tab_help)
                continue

            if typ == "routing":
                ttk.Label(inner, text=label).grid(
                    row=row, column=0, sticky="w", padx=2, pady=(6, 2))
                badge = self._badge(inner, row)
                row += 1
                cur = value if isinstance(value, dict) else {}
                opts = spec.get("choices") or ["auto", "local", "cloud"]
                rvars = {}
                for fn in spec.get("default", {}):
                    ttk.Label(inner, text=f"    • {fn}").grid(
                        row=row, column=0, sticky="w", padx=12, pady=(0, 2))
                    rvar = tk.StringVar(
                        value=str(cur.get(fn, spec["default"][fn])))
                    rvars[fn] = rvar
                    self._combo(inner, rvar, opts, True, scroll_page).grid(
                        row=row, column=1, sticky="we", padx=2, pady=(0, 2))
                    row += 1

                def _get(rv=rvars):
                    return {fn: v.get() for fn, v in rv.items()}

                def _set(val, rv=rvars):
                    for fn, v in rv.items():
                        if isinstance(val, dict) and fn in val:
                            v.set(val[fn])
                f = _Field(key, spec, _get, _set)
                f.badge_label = badge
                f.routing_vars = rvars
                self._register(f, None)
                for fn, rvar in rvars.items():
                    self._trace(rvar, lambda *_a, k=key: self._on_change(k))
                    if f"{key}::{fn}" in VRAM_WATCH_KEYS:
                        self._trace(rvar, lambda *_a: self.update_budget())
                row = self._row_labels(inner, f, row, tab_help)
                continue

            if typ == "device":
                row = self._build_device_row(inner, key, spec, row, tab_help,
                                             scroll_page)
                continue

            # enum / combo / str / int / float → label + control on one row.
            ttk.Label(inner, text=label).grid(
                row=row, column=0, sticky="w", padx=2, pady=(6, 2))
            if typ in ("enum", "combo"):
                var = tk.StringVar(value="" if value is None else str(value))
                values = list(spec.get("choices") or [])
                cur_val = var.get()
                if typ == "combo" and cur_val and cur_val not in values:
                    values = [cur_val] + values   # keep a custom value visible
                if spec.get("suggest") == "monitors":
                    mons = self._monitor_names()
                    if mons:
                        values = ([cur_val] if cur_val and cur_val not in mons
                                  else []) + mons
                combo = self._combo(inner, var, values, typ == "enum",
                                    scroll_page)
                combo.grid(row=row, column=1, sticky="we", padx=2, pady=(6, 2))
                f = _Field(key, spec, var.get, var.set)
                f.combo = combo
            else:
                var = tk.StringVar(value="" if value is None else str(value))
                entry_kw = {}
                if spec.get("secret"):
                    entry_kw["show"] = "•"
                entry = tk.Entry(inner, textvariable=var, bg=FIELD_BG, fg=FG,
                                 insertbackground=FG, font=FONT,
                                 relief="flat", **entry_kw)
                entry.grid(row=row, column=1, sticky="we", padx=2, pady=(6, 2))
                f = _Field(key, spec, var.get, var.set)
                if spec.get("secret"):
                    show = tk.BooleanVar(value=False)
                    ttk.Checkbutton(
                        inner, text="show", variable=show,
                        command=lambda e=entry, s=show: e.configure(
                            show="" if s.get() else "•")).grid(
                        row=row, column=2, sticky="w", padx=(6, 2))
            if not spec.get("secret"):
                f.badge_label = self._badge(inner, row)
            self._register(f, var)
            row += 1
            row = self._row_labels(inner, f, row, tab_help)

        self.notebook.add(outer, text=TAB_LABELS[tab_key])

    def _monitor_names(self) -> list[str]:
        try:
            mons = (self.config_lookup or _config_value)("MONITORS")
        except Exception:
            return []
        if isinstance(mons, dict):
            return [str(k) for k in mons]
        return []

    def _trace(self, var, callback) -> None:
        try:
            var.trace_add("write", callback)
        except Exception:
            pass

    def _register(self, field: _Field, var) -> None:
        field.initial_raw = field.get_raw()
        self.fields[field.key] = field
        if var is not None:
            self._trace(var, lambda *_a, k=field.key: self._on_change(k))
            if field.key in VRAM_WATCH_KEYS:
                self._trace(var, lambda *_a: self.update_budget())

    def _build_device_row(self, parent, key, spec, row, tab_help,
                          scroll_page) -> int:
        tk, ttk = self.tk, self.ttk
        direction = spec.get("direction", "input")
        names_key = spec.get("names_key") or DIRECTION_KEYS[direction][1]
        devices = hostapis = None
        if self._audio_devices is not None:
            devices, hostapis = self._audio_devices
        preferred = self.values.get(names_key) or []
        choices, initial = audio_device_choices(
            direction, self.values.get(key), preferred,
            devices=devices, hostapis=hostapis)
        by_label = {c["label"]: c for c in choices}
        ttk.Label(parent, text=spec.get("label", key)).grid(
            row=row, column=0, sticky="w", padx=2, pady=(6, 2))
        var = tk.StringVar(value=initial["label"])
        combo = self._combo(parent, var, [c["label"] for c in choices], True,
                            scroll_page)
        combo.grid(row=row, column=1, sticky="we", padx=2, pady=(6, 2))
        state = {"owned": initial["value"] if initial["kind"] == "name"
                 else None, "choice": initial, "names_key": names_key,
                 "by_label": by_label, "var": var}
        self.device_pickers[key] = state

        def _get(st=state):
            return device_choice_index(st["by_label"].get(st["var"].get(),
                                                          st["choice"]))

        def _set(v, st=state):
            for c in st["by_label"].values():
                if device_choice_index(c) == v and c["kind"] != "name":
                    st["var"].set(c["label"])
                    return
        f = _Field(key, spec, _get, _set)
        f.combo = combo
        f.badge_label = self._badge(parent, row)
        self._register(f, None)

        def _on_pick(*_a, st=state, k=key):
            choice = st["by_label"].get(st["var"].get())
            if choice is None:
                return
            names_field = self.fields.get(st["names_key"])
            if names_field is not None:
                cur = coerce_value(SCHEMA[st["names_key"]],
                                   names_field.get_raw())
                new = device_choice_list(choice, st["owned"], cur)
                if new != cur:
                    names_field.set_raw(new)
            if choice["kind"] == "name":
                st["owned"] = choice["value"]
            elif choice["kind"] == "auto":
                st["owned"] = None
            st["choice"] = choice
            self._on_change(k)
        self._trace(var, _on_pick)
        return self._row_labels(parent, f, row + 1, tab_help)

    def _build_vram_panel(self, parent, row: int, tab_help) -> int:
        tk, ttk = self.tk, self.ttk
        ttk.Label(parent, text="GPU / VRAM budget", style="Head.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", padx=2, pady=(2, 2))
        row += 1
        if self.vram is None:
            self._help(parent, "(VRAM estimate unavailable — core.vram_budget "
                               "could not load.)", row, tab_help)
            return row + 1
        bar = tk.Canvas(parent, height=18, bg=FIELD_BG, highlightthickness=1,
                        highlightbackground=BORDER, bd=0)
        bar.grid(row=row, column=0, columnspan=3, sticky="we", padx=2,
                 pady=(0, 2))
        fill = bar.create_rectangle(0, 0, 0, 18, fill=OK_GREEN, width=0)
        # Over budget only (see vram_bar_layout): the overage, hatched, past a
        # limit mark where the budget runs out, and a "+N GB over" label.
        over = bar.create_rectangle(0, 0, 0, 18, fill=ERROR, width=0,
                                    stipple="gray50", state="hidden")
        limit = bar.create_line(0, 0, 0, 18, fill=FG, width=2, state="hidden")
        over_text = bar.create_text(0, 9, text="", fill=FG, font=FONT_SMALL,
                                    anchor="e", state="hidden")
        self.vram_widgets["canvas"] = bar
        self.vram_widgets["bar_fill"] = fill
        self.vram_widgets["bar_over"] = over
        self.vram_widgets["bar_limit"] = limit
        self.vram_widgets["bar_over_text"] = over_text
        bar.bind("<Configure>", lambda *_a: self.update_budget())
        row += 1
        num = tk.Label(parent, text="", bg=BG, fg=FG, font=FONT, anchor="w")
        num.grid(row=row, column=0, columnspan=3, sticky="w", padx=2)
        self.vram_widgets["num"] = num
        row += 1
        parts = tk.Label(parent, text="", bg=BG, fg=MUTED, font=FONT_SMALL,
                         anchor="w", justify="left",
                         wraplength=int(700 * self.scale))
        parts.grid(row=row, column=0, columnspan=3, sticky="we", padx=2,
                   pady=(0, 2))
        tab_help.append(parts)
        self.vram_widgets["parts"] = parts
        row += 1
        warn = tk.Label(parent, text="", bg=BG, fg=ERROR, font=FONT_SMALL,
                        anchor="w", justify="left",
                        wraplength=int(700 * self.scale))
        warn.grid(row=row, column=0, columnspan=3, sticky="we", padx=2,
                  pady=(0, 6))
        warn.grid_remove()
        tab_help.append(warn)
        self.vram_widgets["warn"] = warn
        return row + 1

    # ── live behaviour ────────────────────────────────────────────────
    def widget_values(self) -> dict:
        """Current value of every field (the snapshot for fields that don't
        parse yet), flattened MODEL_ROUTING::fn included — what the budget and
        the effective-state notes are computed from."""
        out = dict(self.values)
        for key, f in self.fields.items():
            try:
                raw = f.get_raw()
            except Exception:
                continue
            if raw == f.initial_raw:
                out[key] = self.snapshot.get(key)
                continue
            val, err = validate_value(f.spec, raw)
            out[key] = self.snapshot.get(key) if err else val
        routing = out.get("MODEL_ROUTING")
        if isinstance(routing, dict):
            for fn, v in routing.items():
                out[f"MODEL_ROUTING::{fn}"] = v
        return out

    def _on_change(self, key: str) -> None:
        f = self.fields.get(key)
        if f is not None and f.error_label is not None:
            try:
                f.error_label.configure(text="")
                f.error_label.grid_remove()
            except Exception:
                pass
        self._refresh_warnings()

    def _refresh_warnings(self) -> None:
        notes = effective_warnings(self.widget_values(),
                                   find_spec=self._find_spec)
        for key, f in self.fields.items():
            if f.note_label is None:
                continue
            text = notes.get(key, "")
            try:
                f.note_label.configure(text=("⚠ " + text) if text else "")
                if text:
                    f.note_label.grid()
                else:
                    f.note_label.grid_remove()
            except Exception:
                pass

    def _refresh_badges(self) -> None:
        running = self.pending_running or set()
        for key, f in self.fields.items():
            if f.badge_label is None:
                continue
            if key in self.saved_this_session:
                text = "saved · restart to apply"
            elif key in running:
                text = "saved · not running yet"
            else:
                text = ""
            try:
                f.badge_label.configure(text=text)
            except Exception:
                pass

    def update_budget(self, *_a) -> None:
        """Recompute the VRAM prediction from the live widget values and
        repaint the bar, numbers, breakdown and warning. Never raises."""
        if not self.vram_widgets:
            return
        try:
            wv = self.widget_values()
            widget = {k: wv[k] for k in VRAM_WATCH_KEYS if k in wv}
            b = budget_from_live_values(resolve_vram_values(widget, self.values),
                                        total_mb=self.vram_total_mb)
            if b is None:
                return
            pct = b["pct"]
            color = ERROR if (b["over"] or pct > 100.0) else (
                WARN if pct >= 80.0 else OK_GREEN)
            canvas = self.vram_widgets.get("canvas")
            if canvas is not None:
                try:
                    cw = max(1, int(canvas.winfo_width() or 0))
                except Exception:
                    cw = 1
                if cw <= 1:
                    cw = int(700 * self.scale)
                lay = vram_bar_layout(b, cw)
                canvas.coords(self.vram_widgets["bar_fill"], 0, 0,
                              lay["fill_px"], 18)
                canvas.itemconfigure(self.vram_widgets["bar_fill"], fill=color)
                over = self.vram_widgets["bar_over"]
                limit = self.vram_widgets["bar_limit"]
                over_text = self.vram_widgets["bar_over_text"]
                x = lay["limit_px"]
                if x is not None:
                    canvas.coords(over, x, 0, cw, 18)
                    canvas.coords(limit, x, 0, x, 18)
                    canvas.coords(over_text, cw - 4, 9)
                    canvas.itemconfigure(over_text, text=lay["over_label"])
                for item in (over, limit, over_text):
                    canvas.itemconfigure(
                        item, state="hidden" if x is None else "normal")
            used = b["total_mb"] / 1024.0
            cap = b["total_card_mb"] / 1024.0
            self.vram_widgets["num"].configure(
                text=f"{used:.1f} / {cap:.0f} GB peak  ({pct:.0f}%)", fg=color)
            self.vram_widgets["parts"].configure(text=budget_parts_text(b))
            warn = self.vram_widgets.get("warn")
            if warn is not None:
                if b["over"]:
                    warn.configure(text=self.vram.over_warning(b))
                    warn.grid()
                else:
                    warn.configure(text="")
                    warn.grid_remove()
        except Exception:
            pass

    def collect(self) -> tuple[dict, dict]:
        """``(changes, errors)``: every field whose widget no longer holds the
        value it opened with, validated. A field the owner did not touch is
        never written — nor validated (a bad value already in the file must
        not block saving an unrelated field)."""
        changes: dict = {}
        errors: dict = {}
        for key, f in self.fields.items():
            try:
                raw = f.get_raw()
            except Exception as exc:
                errors[key] = str(exc)
                continue
            if raw == f.initial_raw:
                continue
            value, err = validate_value(f.spec, raw)
            if err:
                errors[key] = err
                continue
            if not values_equal(value, self.snapshot.get(key)):
                changes[key] = value
        return changes, errors

    def _show_errors(self, errors: dict) -> None:
        for key, f in self.fields.items():
            if f.error_label is None:
                continue
            msg = errors.get(key)
            try:
                if msg:
                    f.error_label.configure(
                        text=f"✖ {f.spec.get('label', key)}: {msg}")
                    f.error_label.grid()
                else:
                    f.error_label.configure(text="")
                    f.error_label.grid_remove()
            except Exception:
                pass
        if errors:
            first = next(iter(errors))
            tab = SCHEMA.get(first, {}).get("tab")
            if tab in TAB_ORDER:
                try:
                    self.notebook.select(TAB_ORDER.index(tab))
                except Exception:
                    pass

    def save(self, restart: bool = False) -> bool:
        """Save the changed fields (and, with ``restart``, ask JARVIS to
        restart). Returns True when nothing went wrong."""
        if self.file_error:
            self.status_var.set("Save disabled — the settings file can't be "
                                "read (see the message at the top).")
            return False
        changes, errors = self.collect()
        self._show_errors(errors)
        if errors:
            n = len(errors)
            self.status_var.set(f"Not saved — fix the {n} field"
                                f"{'s' if n != 1 else ''} marked ✖.")
            return False
        msg = "Nothing changed."
        if changes:
            try:
                _doc, (tag, reason) = save_changed_settings(changes, self.path)
            except SettingsFileError as exc:
                self.file_error = str(exc)
                self._refresh_banner()
                self._sync_save_buttons()
                self.status_var.set("Not saved — the settings file can't be "
                                    "read.")
                return False
            except Exception as exc:
                try:
                    self.mb.showerror("JARVIS Settings",
                                      f"Could not save settings:\n{exc}")
                except Exception:
                    pass
                self.status_var.set("Save failed.")
                return False
            for key, value in changes.items():
                self.snapshot[key] = _copy_value(value)
                self.values[key] = _copy_value(value)
                f = self.fields.get(key)
                if f is not None:
                    f.initial_raw = f.get_raw()
                self.saved_this_session.add(key)
            n = len(changes)
            msg = f"Saved {n} change{'s' if n != 1 else ''}."
            if tag:
                vf = self.fields.get("LOCAL_VISION_MODEL")
                if vf is not None:
                    vf.set_raw(tag)
                    vf.initial_raw = vf.get_raw()
                self.snapshot["LOCAL_VISION_MODEL"] = tag
                self.values["LOCAL_VISION_MODEL"] = tag
                self.saved_this_session.add("LOCAL_VISION_MODEL")
                msg += f" Vision model moved with the brain → {tag}."
            elif reason == LOCKSTEP_TEXT_ONLY_REASON:
                vis = self.values.get("LOCAL_VISION_MODEL") or "its own model"
                msg += (f" Local vision stays on {vis} — the new chat model "
                        f"isn't vision-capable.")
            self._refresh_badges()
            self._refresh_warnings()
        if restart:
            if send_tray_command("restart", path=self._command_path):
                self.saved_this_session.clear()
                self._refresh_badges()
                msg += " Restart requested — JARVIS restarts within seconds " \
                       "if it is running."
            else:
                msg += " Could not send the restart request."
                self.status_var.set(msg)
                return False
        elif changes:
            msg += " Takes effect when JARVIS restarts."
        self.status_var.set(msg)
        return True

    def has_unsaved_changes(self) -> bool:
        changes, errors = self.collect()
        return bool(changes or errors)

    def close(self) -> None:
        if self.closed:
            return
        if not self.file_error and self.has_unsaved_changes():
            try:
                ans = self.mb.askyesnocancel(
                    "JARVIS Settings", "Save your changes before closing?",
                    parent=self.root)
            except Exception:
                ans = False
            if ans is None:
                return
            if ans and not self.save():
                return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def open_json(self) -> None:
        try:
            if not os.path.exists(self.path):
                ensure_settings_file(self.path)
            opener = self._file_opener or getattr(os, "startfile", None)
            if opener is None:
                self.status_var.set(self.path)
                return
            opener(self.path)
            self.status_var.set("Opened user_settings.json — reopen this "
                                "window after editing it by hand.")
        except Exception as exc:
            self.status_var.set(f"Open failed: {exc}")

    def raise_window(self) -> None:
        """Bring the window to the front WITHOUT pinning it there (it used to
        be always-on-top, which covered the editor "Open user_settings.json"
        started)."""
        root = self.root
        try:
            root.deiconify()
            root.lift()
            root.attributes("-topmost", True)
            root.after(400, lambda: root.attributes("-topmost", False))
            root.focus_force()
        except Exception:
            pass

    # ── background work, polled on the Tk thread ──────────────────────
    def _start_async(self, name: str, fn) -> None:
        slot = {"done": False, "result": None}
        self._async[name] = slot

        def _run():
            try:
                slot["result"] = fn()
            except Exception:
                slot["result"] = None
            slot["done"] = True
        threading.Thread(target=_run, name=f"settings-{name}",
                         daemon=True).start()

    def _apply_async(self) -> None:
        models = self._async.get("models")
        if models and models["done"] and not models.get("applied"):
            models["applied"] = True
            chat, vision = (models["result"] or (None, None))
            for key, spec in SCHEMA.items():
                f = self.fields.get(key)
                if f is None or f.combo is None:
                    continue
                src = spec.get("suggest")
                if src == "ollama" and chat:
                    vals = list(chat)
                elif src == "ollama-vision":
                    vals = list(vision or chat or [])
                    if not vals:
                        continue
                    vals = vals + (["off"] if "off" not in vals else [])
                else:
                    continue
                cur = str(f.get_raw() or "")
                if cur and cur not in vals:
                    vals = [cur] + vals
                try:
                    f.combo.configure(values=vals)
                except Exception:
                    pass
        running = self._async.get("running")
        if running and running["done"] and not running.get("applied"):
            running["applied"] = True
            if isinstance(running["result"], set):
                self.pending_running = running["result"]
                self._refresh_badges()

    def _tick(self) -> None:
        if self.closed:
            return
        try:
            self._apply_async()
            if self._focus_tag and consume_focus_request(self._focus_tag,
                                                         self._focus_dir):
                self.raise_window()
        except Exception:
            pass
        try:
            self.root.after(self._poll_ms, self._tick)
        except Exception:
            pass


def run_gui(start_tab: int = 0, focus_tag: str | None = None) -> int:
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
    try:
        app = SettingsApp(tk, ttk, messagebox, start_tab=start_tab,
                          focus_tag=focus_tag)
    except Exception as exc:  # pragma: no cover - no display
        sys.stderr.write(f"settings_window: could not open ({exc})\n")
        return 2
    try:
        app.root.mainloop()
    finally:
        try:
            app.root.destroy()
        except Exception:
            pass
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        report = selftest()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    tag = instance_tag()
    handle = acquire_single_instance(tag)
    if handle is None:
        # Another window is already open: bring it forward instead.
        request_focus(tag)
        return 0
    try:
        write_pid_file(tag)
        set_dpi_awareness()
        return run_gui(resolve_start_tab(args.tab), focus_tag=tag)
    finally:
        try:
            os.remove(_pid_file(tag))
        except OSError:
            pass
        release_single_instance(handle)


if __name__ == "__main__":
    raise SystemExit(main())
