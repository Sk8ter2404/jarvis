"""core/prompt_router.py — dynamic system-prompt slimming for the LOCAL brain.

WHY THIS EXISTS
---------------
build_system_prompt() ships ~30k tokens every turn (identity + the 100k-char
PC_CONTROL_PROMPT + skill examples + phrasebook). But the local model's context
is capped at 12–16k tokens by _local_num_ctx() to fit the 3090 — so on the LOCAL
path the prompt is TRUNCATED and the brain never sees its own identity or ~half
its action grammar. (Cloud/Claude has 200k ctx and is unaffected — this module
is LOCAL-only.)

PC_CONTROL_PROMPT is already sectioned by capability (MUSIC CONTROLS, TIMERS /
REMINDERS, BAMBU 3D PRINTER, TTS BACKEND SWITCHING, …). Most turns need one or
two of them. This module keeps the always-relevant CORE preamble, adds only the
sections a turn's text actually implicates, and appends a one-line INDEX of the
rest so the model still KNOWS those capabilities exist (and a follow-up turn can
pull the full section). Net: ~30k → ~6k tokens, so the full relevant instruction
set fits the window uncut, the KV cache shrinks from ~9GB to ~2GB (freeing VRAM
for a bigger brain), and answers sharpen (no lost-in-the-middle over 30k tokens).

Deterministic keyword routing (no model call, no latency). Conservative: when in
doubt it INCLUDES a section, and the INDEX is a safety net for anything dropped.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# A section header in PC_CONTROL_PROMPT: an ALL-CAPS "head" at column 0, ending
# in ':', OPTIONALLY followed by a lowercase parenthetical BEFORE the colon.
# The head is captured as the section name; the parenthetical is descriptive only.
#   "MUSIC CONTROLS:"                                   -> head "MUSIC CONTROLS"
#   "BAMBU 3D PRINTER (H2D):"                           -> head "BAMBU 3D PRINTER"
#   "SMART HOME (router across Hue / Govee / LIFX ...):"-> head "SMART HOME"
#   "SELF-PRESERVATION (CRITICAL — read carefully):"    -> head "SELF-PRESERVATION"
# CRITICAL: the old regex required the WHOLE line to be uppercase, so it matched
# only 12 of the ~54 real headers — the other ~42 capability blocks were silently
# folded into the preceding matched section (bloating it) and vanished from the
# INDEX safety net. The parenthetical is the dominant header style in prompts.py,
# so tolerating it is load-bearing, not cosmetic (2026-07-15 review finding).
_HEADER_RE = re.compile(r"^(?P<head>[A-Z][A-Z0-9 +/&.'\-—]{2,60})(?:\s*\([^)]*\))?:\s*$")

# A header whose descriptive parenthetical WRAPS onto the next line(s) can't be
# seen by the single-line _HEADER_RE, so that header AND its whole capability
# block get silently folded into the previous section and vanish from the INDEX
# safety net. 14 of ~69 real headers in PC_CONTROL_PROMPT wrap this way
# (2026-07-15 review). _join_wrapped_headers stitches such a header back onto one
# physical line BEFORE matching. A header start = an ALL-CAPS-ish head at the
# line start immediately followed by '(' whose ')' has not closed on that line.
_HEADER_START_RE = re.compile(r"^[A-Z][A-Z0-9 +/&.'\-—]{1,60}\(")


def _join_wrapped_headers(lines: List[str]) -> List[str]:
    """Fold a wrapped-parenthetical header (head + '(' … ')':' spanning up to a
    few lines) into a single line. Conservative: only fires when the '(' is left
    open on a head-looking line and a following line (within 3) closes it with
    '):'; bails on a blank line. Non-header text is returned untouched."""
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if (_HEADER_START_RE.match(s) and not s.endswith("):")
                and s.count("(") > s.count(")")):
            joined = ln.rstrip()
            j, closed = i + 1, False
            while j < n and (j - i) <= 3:
                nxt = lines[j].strip()
                if not nxt:            # blank line ⇒ not a wrapped header
                    break
                joined += " " + nxt
                if nxt.endswith("):"):
                    closed = True
                    j += 1
                    break
                j += 1
            if closed:
                out.append(joined)
                i = j
                continue
        out.append(ln)
        i += 1
    return out

# Which lowercase keywords pull in each section. Keyed by the section header text
# (without the trailing colon, upper-cased) — matched leniently by substring of
# the header so exact punctuation need not match. A turn includes a section if
# ANY of its keywords appears in the (lowercased) user text. Keep these generous:
# a false include costs a few hundred tokens; a false exclude is caught by the
# INDEX. Sections with no entry here are treated as niche (index-only unless the
# header words themselves appear).
_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "MULTI-MONITOR APP LAUNCHING": [
        "open", "launch", "start", "app", "window", "monitor", "screen",
        "chrome", "browser", "code", "vscode", "notepad", "explorer", "move",
        "maximize", "minimize", "close", "switch to", "bring up", "pull up",
    ],
    "WINDOW MANAGEMENT": [
        "window", "move", "resize", "snap", "tile", "maximize", "minimize",
        "restore", "left monitor", "right monitor", "fullscreen", "arrange",
    ],
    "SCREEN VISION": [
        "screen", "what's on", "whats on", "looking at", "read the screen",
        "what do you see", "on my screen", "on screen", "see the screen",
    ],
    "WEBCAM AWARENESS": [
        "camera", "webcam", "see me", "can you see", "pointed at me",
        "look at me", "how do i look",
    ],
    "UNIFIED": ["all cameras", "every camera", "both cameras"],
    "FACE RECOGNITION": [
        "who am i", "recognize", "who is at", "who's at", "face",
        "identify me", "who am i talking",
    ],
    "POINT-TO-CONTROL": ["point", "pointing", "that device", "turn that on", "aim at"],
    "AIR-MOUSE": ["air mouse", "air-mouse", "drive the cursor", "hand mouse", "cursor with my"],
    "GUARD MODE": ["guard", "security", "intruder", "watch the room", "arm the cameras", "guard mode"],
    "MUSIC CONTROLS": [
        "music", "play", "song", "track", "album", "artist", "playlist",
        "spotify", "apple music", "pause", "resume", "skip", "next", "previous",
        "shuffle", "volume", "louder", "quieter", "youtube", "netflix", "tv",
        "movie", "watch", "stream", "put on", "listen",
    ],
    "AUDIO OUTPUT DEVICE": [
        "headset", "headphones", "speakers", "output device", "switch audio",
        "audio output", "play through", "sound through",
    ],
    "LOCAL MODEL SELECTION": [
        "model", "which model", "local model", "your brain", "ollama", "llm",
        "what model", "running locally",
    ],
    "BARGE-IN": ["interrupt", "barge", "stop talking", "cut you off"],
    "TIMERS / REMINDERS": [
        "timer", "remind", "reminder", "alarm", "wake me", "in a minute",
        "minutes", "seconds", "hour", "countdown", "set a", "alert me",
    ],
    "TEAMS CALL SCREENING": [
        "teams", "call screening", "screen my calls", "screen calls", "meeting",
    ],
    "CLAUDE CREDITS": [
        "claude", "credit", "credits", "api", "quota", "budget", "cost",
        "spending", "usage", "token",
    ],
    "SYSTEM HEALTH": [
        "health", "cpu", "gpu", "ram", "memory usage", "disk", "temperature",
        "temp", "status", "diagnostic", "diagnostics", "how are you running",
        "system", "load", "vram", "fans", "hardware",
    ],
    "BAMBU 3D PRINTER": [
        "print", "printer", "printing", "bambu", "3d", "filament", "nozzle",
        "bed", "spool", "ams", "h2d", "gcode", "slice",
    ],
    "MORNING BRIEFING": [
        "briefing", "brief me", "morning briefing", "good morning", "my day",
        "agenda", "what's on today",
    ],
    "NEWS BRIEFING": ["news", "headlines", "what's happening", "current events"],
    "DAILY RECAP": [
        "recap", "daily recap", "end of day", "summary of my day", "how was my day",
    ],
    "DOSSIER": [
        "dossier", "pull up the file", "file on", "what do you know about",
        "tell me about",
    ],
    "SUIT-UP CINEMATIC": ["suit up", "suit-up", "boot sequence", "cinematic"],
    "TASK QUEUE": [
        "task queue", "queue this", "offload", "claude code", "add a task",
        "todo", "to-do", "build me", "have claude",
    ],
    "SESSION MEMORY RECALL": [
        "remember", "recall", "what did", "earlier", "last time", "before",
        "you said", "we talked", "memory", "forget", "note that", "keep in mind",
    ],
    "SESSION RESUME": [
        "resume", "continue", "where were we", "pick up", "carry on",
        "last session", "what were we",
    ],
    "DO-NOT-DISTURB FOCUS MODE": [
        "focus", "do not disturb", "dnd", "quiet mode", "silence", "mute me",
        "concentrate", "no interruptions", "leave me alone",
    ],
    "NIGHT-OWL MODE": ["night owl", "late night", "wind down", "dim", "night mode"],
    "SELF-PRESERVATION": [
        "shut yourself", "kill you", "turn you off", "stay online",
        "don't shut down", "preserve yourself", "shut you down",
    ],
    "UI AUTOMATION": [
        "click on", "type into", "automate", "fill in", "press the button",
        "move the mouse", "click the",
    ],
    "CHANGELOG / VERSION": [
        "version", "what's new", "changelog", "update notes", "your version",
        "what changed", "what version",
    ],
    "SKILLS": ["learn", "teach yourself", "new skill", "teach you", "can you learn"],
    "SMART HOME": [
        "light", "lights", "hue", "govee", "lifx", "kasa", "ecobee", "nest",
        "thermostat", "plug", "bulb", "dim", "brighten", "turn on the",
        "turn off the", "smart home", "lamp",
    ],
    "NETWORK / LAN PRESENCE": [
        "network", "wifi", "wi-fi", "router", "deco", "who's home", "whos home",
        "devices online", "lan", "is home", "internet",
    ],
    "OBS STUDIO": [
        "obs", "record", "recording", "stream", "streaming", "scene",
        "start recording",
    ],
    "PYTHON SANDBOX": [
        "python", "calculate", "run code", "run a script", "compute", "evaluate",
        "what's the square", "math",
    ],
    "IMAGE GENERATION": [
        "generate an image", "make a picture", "draw me", "image of",
        "create an image", "sdxl", "picture of", "generate a picture",
    ],
    "LOCAL VISION": ["offline vision", "local vision", "vlm"],
    "PERSONAL RAG": [
        "my notes", "my files", "search my", "my documents", "in my files",
        "find in my", "my docs", "my notes about",
    ],
    "TTS BACKEND SWITCHING": [
        "voice", "tts", "speak like", "sound like", "british", "accent",
        "switch voice", "your voice", "talk like", "edge", "clone voice",
        "kokoro",
    ],
    "VOICE ENROLLMENT / SPEAKER ID": [
        "enroll", "my voice", "learn my voice", "who am i", "speaker",
        "recognize me", "register my voice", "voice id", "identify me",
    ],
    "PHONE NOTIFICATIONS": [
        "phone", "notify my phone", "telegram", "ntfy", "pushover", "text me",
        "send to my phone", "push to my",
    ],
    "SCHEDULING": [
        "schedule", "every day", "cron", "recurring", "remind me every",
        "trigger when", "when x happens", "each morning", "daily at",
    ],
    "MCP TOOLS": ["mcp", "tool server", "model context protocol"],
    "BROWSER AGENT": [
        "browse", "web automation", "playwright", "go to the website",
        "fill the form", "book a", "order online", "navigate to",
    ],
    "EMAIL TRIAGE": [
        "email", "inbox", "gmail", "outlook", "unread", "mail", "my emails",
    ],
    "SMART HOME DISCOVERY": [
        "discover devices", "find my lights", "scan for devices", "find devices",
    ],
    "TV DETECTION": ["tv", "television", "is the tv"],
    "KINECT GAZE TRACKING": ["gaze", "where am i looking", "eye tracking"],
    "AMAZON ORDER TRACKER": [
        "amazon", "order", "package", "delivery", "tracking", "my orders",
        "where's my package", "wheres my package",
    ],
    "DECO MESH NETWORK": ["deco", "mesh", "router"],
    "NOTIFICATION TRIAGE": ["notification", "notifications", "alerts", "my alerts"],
    "PHONE BRIDGE": ["phone bridge", "my phone"],
    "SELF DIAGNOSTIC": [
        "diagnostic", "health check", "self test", "self-diagnostic",
        "are you ok", "run diagnostics", "check yourself",
    ],
    "STABILITY GATE": ["stability", "safe to upgrade", "stability gate"],
    "WAKE LISTENER": [
        "wake word", "hey jarvis", "porcupine", "stop listening",
        "start listening", "listen for",
    ],
    "CODE EXECUTOR": ["run python", "execute python", "code executor"],
    "CUSTOM TTS / XTTS": ["custom voice", "xtts", "clone a voice", "custom tts"],
    "MCP": ["mcp"],
    "OBS": ["obs"],
    "BAMBU PRINTER LAN CHECK ALIAS": [
        "print", "printer", "bambu", "3d", "is it printing", "printer online",
    ],
    "REPLAY": [
        "replay", "say again", "repeat that", "come again", "what did you say",
        "one more time",
    ],
    "SHUTDOWN ALIASES": [
        "shut down", "shutdown", "turn off", "go offline", "power down",
        "restart yourself", "reboot", "sign off",
    ],
    # --- Sections surfaced by the 2026-07-15 wrapped-header + char-class fix.
    # These 17 were folded into their neighbours (invisible to routing) until the
    # parser learned to read multi-line/punctuated headers; give each real
    # keywords so naming the capability loads its full instructions, not just its
    # INDEX line. Kept specific to avoid taxing unrelated turns.
    "KINECT DEPTH SENSOR": [
        "kinect", "depth sensor", "who is in the room", "who's in the room",
        "scan the room", "scan room", "how many people", "body count",
        "anyone in the room", "who is here", "who's here",
    ],
    "AIR CONTROL": [
        "air control", "spatial mouse", "reach out", "grab and drag",
        "fist grab", "kinect mouse", "movie-style",
    ],
    "MUSIC + VIDEO PLAYBACK": [
        "playback", "play a video", "media keys", "play/pause", "media control",
        "resume playback", "pause playback",
    ],
    "STREAMING SERVICES": [
        "netflix", "hulu", "disney", "disney+", "hbo", "prime video",
        "streaming service", "watch on", "auto-play", "put on a movie",
    ],
    "TASTE-AWARE MUSIC": [
        "my music taste", "recommend a song", "recommend music", "based on my taste",
        "music recommendation", "something i'd like", "music i'd like",
    ],
    "FOCUS MODE / DO-NOT-DISTURB": [
        "focus mode", "do not disturb", "hold my notifications", "heads down",
        "what did i miss", "recap what i missed",
    ],
    "WEB INTERFACE": [
        "web interface", "dashboard", "control panel", "web ui", "web dashboard",
        "open the dashboard", "browser control panel",
    ],
    "WELLNESS / FOCUS NUDGES": [
        "wellness", "focus block", "take a break", "break reminder",
        "focus session", "pomodoro", "stretch reminder", "posture",
    ],
    "CALENDAR": [
        "calendar", "schedule", "my meetings", "appointment", "agenda",
        "meetings today", "meetings this week", "what meetings", "on my calendar",
    ],
    "WEATHER BRIEFING": [
        "weather", "forecast", "is it going to rain", "raining", "sunny",
        "snow", "how hot", "how cold", "umbrella", "outside today",
    ],
    "PATTERN LEARNING": [
        "my patterns", "my habits", "learned about me", "my routine",
        "behavioral pattern", "what have you noticed", "patterns you've",
    ],
    "REPO ROBOT PROJECT": [
        "repo robot", "animatronic", "robot project", "the robot build",
        "robot state",
    ],
    "SUIT DIAGNOSTICS": [
        "suit diagnostics", "full system readout", "full diagnostics",
        "full readout", "complete diagnostics", "detailed diagnostics",
    ],
    "MULTI-STEP TASKS": [
        "add to cart", "find and add", "and add to", "buy me", "order online",
        "checkout", "multi-step", "then click", "do all of",
    ],
    "LOCAL VOICE CLONE": [
        "voice clone", "cloned voice", "voice profile", "clone voice",
        "your own voice", "in-character voice", "chatterbox", "switch to my voice",
    ],
    "SMART HOME — PER-BRAND LIST": [
        "list my lights", "list plugs", "which lights", "hue list", "govee list",
        "kasa list", "per brand", "brand list", "list smart",
    ],
    "WAKE-WORD MODE": [
        "wake word mode", "wake-word mode", "require my name", "require your name",
        "always listening", "manual wake", "gate on wake",
    ],
}

# Sections always kept even with no keyword hit. Deliberately MINIMAL: only
# app-launching (small — 1.3k chars — and the single most fundamental PC-control
# capability). MUSIC (12k chars) and TIMERS (4.5k chars) are large and have
# strong, unambiguous keywords ("play"/"song"/"spotify", "timer"/"remind"), so
# they load exactly when relevant instead of taxing every turn. This keeps the
# common-turn PC block near ~3k tokens so BASE identity + rules + phrasebook all
# fit the 12k window uncut.
_ALWAYS = {"MULTI-MONITOR APP LAUNCHING"}


def split_pc_control(pc_control: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split PC_CONTROL_PROMPT into (core_preamble, [(header, body), ...]).

    core_preamble = everything before the first section header (the general
    action-format rules + intro that must always be present). Each subsequent
    (header, body) pair is one capability section, body INCLUDING the header
    line so the reinjected text is self-describing."""
    lines = _join_wrapped_headers(pc_control.split("\n"))
    first_hdr = None
    for i, ln in enumerate(lines):
        if _HEADER_RE.match(ln.strip()):
            first_hdr = i
            break
    if first_hdr is None:
        return pc_control, []
    core = "\n".join(lines[:first_hdr])
    sections: List[Tuple[str, str]] = []
    cur_name = None
    cur_lines: List[str] = []
    for ln in lines[first_hdr:]:
        m = _HEADER_RE.match(ln.strip())
        if m:
            if cur_name is not None:
                sections.append((cur_name, "\n".join(cur_lines)))
            cur_name = m.group("head").strip()
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    if cur_name is not None:
        sections.append((cur_name, "\n".join(cur_lines)))
    return core, sections


def _keywords_for(header: str) -> List[str]:
    """Keyword list for a section header, tolerant of punctuation differences."""
    key = header.upper().strip()
    if key in _SECTION_KEYWORDS:
        return _SECTION_KEYWORDS[key]
    # tolerant match: compare on alnum-only
    norm = re.sub(r"[^A-Z0-9]", "", key)
    for k, v in _SECTION_KEYWORDS.items():
        if re.sub(r"[^A-Z0-9]", "", k) == norm:
            return v
    return []


def select_sections(user_text: str, sections: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Return (included_section_names, dropped_section_names) for `user_text`."""
    low = " " + (user_text or "").lower() + " "
    included: List[str] = []
    dropped: List[str] = []
    for header, _body in sections:
        name = header.strip()
        upper = name.upper()
        hit = name.upper() in _ALWAYS
        if not hit:
            # header words present in the query?
            words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if len(w) > 3]
            if any(f" {w}" in low or f"{w} " in low for w in words):
                hit = True
        if not hit:
            for kw in _keywords_for(name):
                if kw in low:
                    hit = True
                    break
        (included if hit else dropped).append(name)
    return included, dropped


def slim_pc_control(user_text: str, pc_control: str) -> str:
    """Build a slimmed PC_CONTROL for this turn: core preamble + the sections the
    text implicates + a one-line INDEX of what was left out (so the model still
    knows those capabilities exist). Falls back to the full text if parsing finds
    no sections. Never raises — a bad parse returns the full prompt."""
    try:
        core, sections = split_pc_control(pc_control)
        if not sections:
            return pc_control
        included, dropped = select_sections(user_text, sections)
        inc_set = set(included)
        parts = [core]
        for header, body in sections:
            if header.strip() in inc_set:
                parts.append(body)
        if dropped:
            parts.append(
                "\n\nADDITIONAL CAPABILITIES (ask and I'll use them; full "
                "instructions load on request): " + "; ".join(dropped) + ".")
        return "\n".join(parts)
    except Exception:
        return pc_control
