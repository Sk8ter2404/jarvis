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

# A section header in PC_CONTROL_PROMPT: an ALL-CAPS-ish line ending in ':' at
# (near) column 0. Matches "MUSIC CONTROLS:", "BAMBU 3D PRINTER (H2D):", etc.
_HEADER_RE = re.compile(r"^([A-Z][A-Z0-9 /&()\.\'\-]{3,50}:)\s*$")

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
    "MUSIC CONTROLS": [
        "music", "play", "song", "track", "album", "artist", "playlist",
        "spotify", "apple music", "pause", "resume", "skip", "next", "previous",
        "shuffle", "volume", "louder", "quieter", "youtube", "netflix", "tv",
        "movie", "watch", "stream", "put on", "listen",
    ],
    "TIMERS / REMINDERS": [
        "timer", "remind", "reminder", "alarm", "wake me", "in a minute",
        "minutes", "seconds", "hour", "countdown", "set a", "alert me",
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
    "BAMBU 3D PRINTER (H2D)": [
        "print", "printer", "printing", "bambu", "3d", "filament", "nozzle",
        "bed", "spool", "ams", "h2d", "gcode", "slice",
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
        "night owl", "concentrate", "no interruptions", "leave me alone",
    ],
    "TTS BACKEND SWITCHING": [
        "voice", "tts", "speak like", "sound like", "british", "accent",
        "switch voice", "your voice", "talk like", "edge", "clone voice",
    ],
    "VOICE ENROLLMENT / SPEAKER ID": [
        "enroll", "my voice", "learn my voice", "who am i", "speaker",
        "recognize me", "register my voice", "voice id", "identify me",
    ],
    "BAMBU PRINTER LAN CHECK ALIAS": [
        "print", "printer", "bambu", "3d", "is it printing", "printer online",
    ],
    "REPLAY": [
        "replay", "say again", "repeat that", "come again", "what did you say",
        "one more time",
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
    lines = pc_control.split("\n")
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
            cur_name = m.group(1).rstrip(":").strip()
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
