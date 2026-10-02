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

from core.action_risk import SELF_TERMINATING_ACTIONS
from core.spoken_math import is_arithmetic_request
from core.units import is_unit_conversion_request

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
# ANY of its keywords appears in the (lowercased) user text AT THE START OF A
# WORD (see _keyword_hit: "phone" no longer fires inside "microphone"; a
# keyword may still run on, "print" -> "printing"). Keep these generous:
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
        # Reading requests name the page, not the screen (live 2026-10-01).
        "this page", "the page", "read this", "this article",
    ],
    "WEBCAM AWARENESS": [
        "camera", "webcam", "see me", "can you see", "pointed at me",
        "look at me", "how do i look",
    ],
    # UNIFIED documents camera_status / situational_awareness (where_am_i) /
    # look_around — cover every trigger phrase its body spells out, not just
    # the "all cameras" phrasings (2026-07-21 audit: 'where am I' / 'camera
    # status' / 'look around' loaded nothing).
    "UNIFIED": [
        "all cameras", "every camera", "both cameras", "camera status",
        "what cameras", "where am i", "what am i doing", "what's my status",
        "look around", "see everywhere",
        # 2026-10-01 brain eval (cam-01): 'can you tell where I'm sitting'
        # loaded nothing, so situational_awareness never reached the model.
        "where i'm sitting", "where im sitting", "where i am sitting",
        "am i sitting", "where i'm standing", "where i am standing",
        "am i standing", "how far back am i",
        # 2026-09-06 live regression: "are both webcams ok" and "can you check
        # if both webcams are still working" BOTH emitted system_pulse and
        # answered with CPU/memory percentages, never mentioning a camera. The
        # literal "camera status" routed fine, so camera_status was reachable —
        # the router just failed on paraphrase, because "webcam" lived only in
        # the WEBCAM AWARENESS section and nothing mapped webcam + health here.
        "webcam", "webcams", "cameras working", "camera working",
        "cameras ok", "webcams ok", "cameras still", "webcams still",
        "camera health", "cameras up",
        # 2026-10-01: camera_unquarantine ("use the Kinect again", "use the left
        # webcam again") is documented here, but "kinect" alone routed only to
        # KINECT DEPTH SENSOR, whose kinect_status the model then ran instead.
        "kinect again", "webcam again", "camera again", "cameras again",
        "retry the kinect", "retry the camera",
    ],
    "FACE RECOGNITION": [
        "who am i", "recognize", "who is at", "who's at", "face",
        "identify me", "who am i talking",
    ],
    "POINT-TO-CONTROL": ["point", "pointing", "that device", "turn that on", "aim at"],
    # 2026-09-06 whole-prompt arrow-example audit (the follow-up the STATUS
    # READ-BACKS card named): this section prints "'take the cursor' →
    # [ACTION: air_mouse_arm]" and "'release the cursor' → [ACTION:
    # air_mouse_disarm]" as its own examples, and BOTH selected nothing but the
    # always-on app launcher — so the arm/disarm tokens never reached the local
    # model and the nearest thing it could see was air_mouse_on/off, which is a
    # DIFFERENT operation (feature power vs. the pose gate). Cover every
    # trigger phrase the body prints, not just the "air mouse" spellings.
    "AIR-MOUSE": [
        "air mouse", "air-mouse", "drive the cursor", "hand mouse",
        "cursor with my", "take the cursor", "give me the cursor",
        "release the cursor", "grab the cursor", "mouse control",
        "calibrate reach",
    ],
    "GUARD MODE": ["guard", "security", "intruder", "watch the room", "arm the cameras", "guard mode"],
    "MUSIC CONTROLS": [
        "music", "play", "song", "track", "album", "artist", "playlist",
        "spotify", "apple music", "pause", "resume", "skip", "next", "previous",
        "shuffle", "volume", "louder", "quieter", "youtube", "netflix", "tv",
        "movie", "watch", "stream", "put on", "listen",
        # volume grammar lives in this section's body — "mute" had NO keyword
        # anywhere and the un-anchored turn-it-down/up phrasings only matched
        # the dispatcher's anchored fast-paths (2026-07-21 audit).
        "mute", "turn it down", "turn it up",
        # 2026-10-01: volume_unmute's own example ('turn the sound back on').
        # "unmute" needs its own entry: keywords match at a word start now,
        # so "mute" no longer fires inside it.
        "sound back on", "unmute",
        # 2026-10-01 brain eval (media-05): 'kill the sound completely'
        # loaded nothing, so volume_mute never reached the model. Verb +
        # object only: a bare "sound" would ride every "sounds good".
        "kill the sound", "cut the sound", "kill the audio", "cut the audio",
        "kill the volume", "turn off the sound", "turn the sound off",
    ],
    "AUDIO OUTPUT DEVICE": [
        "headset", "headphones", "speakers", "output device", "switch audio",
        "audio output", "play through", "sound through",
        # 2026-09-06: 'is auto switching on' → audio_autoswitch_status loaded
        # TTS BACKEND SWITCHING (on the bare word "switch") and nothing else —
        # the wrong subsystem entirely, so the model was offered voice-backend
        # tokens for a question about the audio-device WATCHER. "auto switch"
        # is a substring of "auto switching"/"auto switches", so one entry
        # covers the tense variants.
        "auto switch", "auto-switch", "autoswitch", "audio status",
    ],
    "LOCAL MODEL SELECTION": [
        "model", "which model", "local model", "your brain", "ollama", "llm",
        "what model", "running locally",
    ],
    # 2026-10-01 live: "how smart are you compared to Claude Opus 5.5" (asked
    # three times) loaded CLAUDE CREDITS + the three SMART HOME sections and
    # got a joke each time; "how do you compare to other AIs" loaded nothing.
    # Its body is rendered per turn from live values (_RUNTIME_SECTIONS).
    "SELF-KNOWLEDGE": [
        # Only phrasings that are about HIM on their own. A brand ("use
        # Claude", "play Opus by Eric Prydz") or a bare comparison ("compared
        # to last week") is not: review 2026-10-02 found 22 of 25 unrelated
        # probes loading this ~2.1k-char section. "you" + a comparison + an
        # AI name, and "how good are you" (but not "... at chess"), route
        # through is_self_knowledge_request (_PREDICATE_ROUTES).
        "how smart are you", "how smart you are", "how smart is jarvis",
        "how clever are you", "how intelligent are you",
        "how capable are you", "your iq", "smarter than you",
        "dumber than you", "better than you",
        "what runs you", "what powers you", "what are you running on",
        "which model are you", "what model are you",
        "compare yourself", "how do you compare",
        "other jarvis", "jarvis-like", "assistants like you",
    ],
    "BARGE-IN": ["interrupt", "barge", "stop talking", "cut you off"],
    "TIMERS / REMINDERS": [
        "timer", "remind", "reminder", "alarm", "wake me", "in a minute",
        "minutes", "seconds", "hour", "countdown", "set a", "alert me",
    ],
    # Header renamed TEAMS CALL SCREENING -> MICROSOFT TEAMS (2026-08-20): the
    # three screener actions it used to document ship only in the gitignored
    # skills/teams_screener.py, so the tracked prompt now covers the unread
    # sweep alone. Keyword list is unchanged so "screen my calls" still loads
    # the section on the local route. The key MUST match the section head or
    # test_every_section_has_routing_or_is_always strands the section.
    "MICROSOFT TEAMS": [
        "teams", "call screening", "screen my calls", "screen calls", "meeting",
    ],
    "CLAUDE CREDITS": [
        "claude", "credit", "credits", "api", "quota", "budget", "cost",
        "spending", "usage", "token",
        # check_credits' 'check my Anthropic balance' loaded nothing; the
        # running_costs phrasings are spelled out so they never ride on the
        # bare "cost" alone.
        "anthropic", "running cost", "cost to run", "you cost",
        "electricity", "power bill",
    ],
    "SYSTEM HEALTH": [
        "health", "cpu", "gpu", "ram", "memory usage", "disk", "temperature",
        "temp", "status", "diagnostic", "diagnostics", "how are you running",
        "system", "load", "vram", "fans", "hardware",
        # 2026-09-06 live regression: "how hot is my graphics card" loaded only
        # MULTI-MONITOR APP LAUNCHING and WEATHER BRIEFING — SYSTEM HEALTH was
        # DROPPED, so gpu_usage never reached the model and it answered with
        # system_pulse ("GPU running at 33 percent") to a question about
        # TEMPERATURE. "how hot is the GPU" worked, which is what proved it was
        # vocabulary and not the action. Note "hot" also pulled in WEATHER
        # BRIEFING, so the wrong section loaded while the right one did not.
        "graphics card", "graphics", "video card", "how hot", "hot", "degrees",
        "celsius", "throttl", "overheat", "running hot",
        # 2026-09-29: check_system reads C: free space, but 'how much space is
        # left on my C drive' selected NOTHING beyond the always-on launcher
        # ("disk" is the only storage word above). The leading space on
        # " c drive" is deliberate: bare "c drive" is a substring of "musiC
        # DRIVE". "usb" rides here because a USB fault report ("I'm still
        # having USB issues") needs a hardware diagnostic in front of the
        # model, and this is the section that owns one.
        " c drive", " c: drive", "disk space", "drive space", "free space",
        "space left", "space is left", "much space", "storage", "hard drive",
        "ssd", "usb",
        # 2026-10-01 brain eval (sys-02): 'how much video memory is free'
        # loaded MUSIC + VIDEO PLAYBACK and both memory sections on the
        # header words "video" / "memory", but never this one (gpu_usage).
        "video memory", "graphics memory", "gpu memory",
    ],
    "BAMBU 3D PRINTER": [
        "print", "printer", "printing", "bambu", "3d", "filament", "nozzle",
        "bed", "spool", "ams", "h2d", "gcode", "slice",
        # 2026-10-01 brain eval (print-04): the chamber-camera panel
        # (show_printer_camera) is documented here now; 'pull up the chamber
        # cam' names no other printer word.
        "chamber cam",
    ],
    "MORNING BRIEFING": [
        "briefing", "brief me", "morning briefing", "good morning", "my day",
        "agenda", "what's on today",
        # 2026-09-06: this section documents FOUR distinct openers
        # (morning_briefing / morning_handoff / predictive_morning_setup /
        # morning_arrival) and the keyword list only covered the word
        # "briefing". Its own examples 'JARVIS, catch me up' → morning_handoff
        # and 'JARVIS, set up my workspace' → predictive_morning_setup selected
        # nothing at all. "arrival"/"cold open" are spelled out rather than
        # left as a bare "arrival" so package-delivery turns stay clear of this.
        "catch me up", "handoff", "hand off", "workspace",
        "morning apps", "morning arrival", "arrival briefing", "cold open",
    ],
    "NEWS BRIEFING": ["news", "headlines", "what's happening", "current events"],
    "DAILY RECAP": [
        "recap", "daily recap", "end of day", "summary of my day", "how was my day",
    ],
    # 2026-09-29: both header words are generic ("project", "status"), so this
    # section routes ONLY on these. A miss here is the live failure it exists
    # for: "what am I working on" answered from mis-heard learned topics.
    "PROJECT STATUS": [
        "working on", "been doing lately", "been up to", "my projects",
        "project status", "project list", "status of my", "my project",
        "what am i doing lately",
    ],
    "DOSSIER": [
        "dossier", "pull up the file", "file on", "what do you know about",
        "tell me about",
    ],
    # skills/globe.py. Bare "where is" / "where's" deliberately NOT here: they
    # are the package, print, robot-build and where-am-I questions, and the
    # globe must not ride along on those. "show me where" is the globe's own
    # phrasing; "pin" is short, so it takes only a plural ("pins"), never
    # "pinned" / "ping" / "spin".
    "GLOBE": [
        "globe", "show me where", "pin", "on the map", "world map",
    ],
    "SUIT-UP CINEMATIC": ["suit up", "suit-up", "boot sequence", "cinematic"],
    "TASK QUEUE": [
        "task queue", "queue this", "offload", "claude code", "add a task",
        "todo", "to-do", "build me", "have claude",
        # The section body also documents the lifecycle + HUD/overlay actions
        # (restart, hide_hud/show_hud/toggle_hud, arc_reactor, the holographic
        # overlay, workshop_hud, upgrade) — without these keywords a 'restart
        # yourself' turn loaded only the SHUTDOWN aliases and 'hide the HUD'
        # loaded nothing (2026-07-21 audit). Substring hits on shutdown-adjacent
        # turns are harmless: the model just sees restart AND shutdown_jarvis.
        "restart", "reboot", "relaunch", "start over", "hud", "overlay",
        "arc reactor", "reactor", "holo", "holographic", "workshop",
        "upgrade", "apply the changes", "doorless",
        # show_tray (2026-09-30) — multi-word so "stray"/"portray" can't hit.
        "tray icon", "the tray", "system tray",
    ],
    "SESSION MEMORY RECALL": [
        "remember", "recall", "what did", "earlier", "last time", "before",
        "you said", "we talked", "memory", "forget", "note that", "keep in mind",
        # 2026-09-29, live: "summarize what we talked about today" got a false
        # "I can only recall specific past conversations" decline, and 'recap
        # our conversation' / 'what have we discussed' loaded no recall section
        # at all ("recap" alone reaches only DAILY RECAP, an app-usage summary).
        # session_memory_recall summarises THIS session too — see its body.
        "talked about", "discussed", "our conversation", "this conversation",
        # 2026-10-01 brain eval (mem-01): 'sum up what we've been chatting
        # about today' loaded nothing — "talked about" has no other tense.
        "chatting about", "chatted about", "been chatting",
        "been talking about", "we've been talking", "we were talking",
        "sum up what we", "sum up our",
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
    # NOTE this key routes TWO physically separate sections — the prompt has a
    # full "CHANGELOG / VERSION (self-awareness…)" block and a short
    # "CHANGELOG / VERSION (additional phrasings)" one, and _keywords_for is
    # keyed by header NAME, so one list serves both. That is fine here (same
    # topic) but it is why the list has to cover the big block's actions too.
    "CHANGELOG / VERSION": [
        "version", "what's new", "changelog", "update notes", "your version",
        "what changed", "what version",
        # 2026-09-06: the big block also owns check_for_updates, model_costs
        # and report_bug, and the list above covered none of them.
        #   'check for updates'  -> selected BAMBU PRINTER LAN CHECK ALIAS (!)
        #   'which model is cheapest?' -> LOCAL MODEL SELECTION + BROWSER AGENT
        # The second is the expensive near-miss: BROWSER AGENT's find_cheapest
        # is a SHOPPING action, so "which model is cheapest" was one token away
        # from opening a price-comparison browse for an LLM.
        "check for update", "new version", "newer version", "up to date",
        "updates available", "newer you",
        "model cost", "model prices", "each model", "compare models",
        "model is cheapest", "model burn",
    ],
    "SKILLS": ["learn", "teach yourself", "new skill", "teach you", "can you learn"],
    "SMART HOME": [
        # Review 2026-10-02: with "smart" a generic header word, a listing
        # that never says "home" ("list my smart devices") lost all three
        # SMART HOME sections. "smart device" runs on to "devices".
        "smart device",
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
    "WEBSITE BUILDER": [
        "website", "web site", "landing page", "web page", "webpage",
        "homepage", "home page", "site for",
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
        # The other two backends this body lists ('switch to xtts' is its own
        # arrow example). They only ever routed by "tts" firing mid-word.
        "xtts", "pyttsx3", "pyttsx",
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
    # 2026-09-04 documented the REST of the browser-agent surface in
    # core/prompts.py (browse_for, find_cheapest, book_appointment, fill_form,
    # browser_status/stop/open/reset_profile) but left this keyword list at its
    # 2026-07 shape, so the section's OWN flagship examples never loaded it.
    # Measured against the live prompt before this fix:
    #   'find me the cheapest 2tb nvme'            -> BROWSER AGENT dropped
    #   'book me a haircut friday afternoon'       -> dropped ("book a" is not
    #                                                 a substring of "book me")
    #   'fill that form in with my name and email' -> dropped ("fill the form"
    #                                                 is not a substring of it)
    #   'go read up on petg nozzle temps'          -> dropped
    # i.e. all four sentences the body prints as "'X' -> [ACTION: Y]" reached
    # the local model as an INDEX line and nothing else. The six phrases that
    # did route only did so because they contain the literal header word
    # "browser", not because of any keyword here.
    # Why this matters more than an ordinary miss: PC_CONTROL_SAFETY_RULES
    # (always shipped, in the core preamble) now tells the model that a
    # near-miss is worse than nothing. With find_cheapest / browse_for /
    # book_appointment / fill_form invisible, the closed list pushes these
    # turns toward "I'm afraid I've no way to check that, sir." rather than the
    # documented action. Keep every quoted trigger the body documents covered
    # here — BrowserAgentRoutingRegressionTests in tests/test_prompt_router.py
    # re-extracts them from the prompt and fails if one stops routing back.
    "BROWSER AGENT": [
        "browse", "web automation", "playwright", "go to the website",
        "fill the form", "book a", "order online", "navigate to",
        # find_cheapest — the body's own example is 'find me the cheapest …',
        # and it is the action that must win over web_search when prices are
        # to be COMPARED across sites.
        "cheapest", "best price", "lowest price", "compare prices",
        "price compare", "shop around", "bargain", "how much is",
        # browse_for — 'go read up on PETG nozzle temps and summarise it'.
        # "look up" is deliberately here even though the body assigns 'look up
        # X' to web_search: that contrast is only teachable if the section is
        # in front of the model when he says it.
        "read up", "look up", "search the web", "on the web", "summarise",
        "summarize", "top three results",
        # book_appointment — "book a" alone missed 'book me a haircut'.
        "book me", "book an", "book the", "appointment", "reservation",
        "reserve a", "reserve me", "make me a booking",
        # fill_form — "fill the form" alone missed 'fill that form in with …'.
        # Word-initial "fill" variants only; a bare "form" would fire on
        # "information" / "format" / "platform" / "perform".
        "fill in", "fill out", "fill that", "fill this", "fill my",
        "fill the",
        # browser_status — 'how's the browser doing' / 'what's it up to'. The
        # first carries the header word; the second carries nothing at all.
        "what's it up to", "whats it up to",
    ],
    "EMAIL TRIAGE": [
        "email", "inbox", "gmail", "outlook", "unread", "mail", "my emails",
    ],
    "SMART HOME DISCOVERY": [
        "smart device", "my smart", "discover smart", "scan for smart",
        "find smart",
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
        # 2026-10-01 (09-05 live diagnostic): 'run a system check' loaded only
        # SYSTEM HEALTH, whose quick CPU/RAM readout then answered it - this
        # section, home of system_check (the full sweep), had no such entry.
        "system check", "systems check", "self check", "self-check",
        "are you alright", "are you all right", "is everything ok",
        "is everything okay",
    ],
    "STABILITY GATE": ["stability", "safe to upgrade", "stability gate"],
    # 2026-09-29: bare "wake word" used to live here, so every WAKE-WORD MODE
    # turn ('turn off wake word mode', 'require the wake word') also shipped
    # wake_listener_start/stop -- a different subsystem one token away. The
    # engine-shaped phrasings below are what this section's body documents.
    "WAKE LISTENER": [
        "wake listener", "wake word listener", "wake word detector",
        "wake word engine", "hotword", "hey jarvis", "porcupine",
        "stop listening", "start listening", "listen for",
    ],
    "CODE EXECUTOR": ["run python", "execute python", "code executor"],
    "CUSTOM TTS / XTTS": ["custom voice", "xtts", "clone a voice", "custom tts"],
    "MCP": ["mcp"],
    "OBS": ["obs"],
    "BAMBU PRINTER LAN CHECK ALIAS": [
        "print", "printer", "bambu", "3d", "is it printing", "printer online",
    ],
    # Only ACTION-replay phrasing (2026-10-01). "say again" / "repeat that" /
    # "come again" / "what did you say" / "one more time" ask JARVIS to SAY
    # something again (tone_detector.is_repeat_request), yet they loaded the
    # one section that documents replay_last_action - steering the model to
    # re-run the last action when the owner only wanted the sentence again.
    "REPLAY": [
        "replay", "do that again", "do it again",
    ],
    # "restart yourself"/"reboot" deliberately NOT here: those phrases mean
    # RESTART, and this section documents only power-off aliases — loading it
    # as the sole power-related section pulled the model toward shutdown_jarvis
    # for a restart request (2026-07-21 audit). TASK QUEUE (which documents
    # `restart`) now carries those keywords.
    # Bare "turn off" is deliberately NOT here (2026-09-29): it matched EVERY
    # "turn off the lights / the web dashboard / wake word mode" turn and put
    # turn_off_jarvis -- a power-off of JARVIS himself -- next to the action
    # the owner meant. Only the self-directed forms route here.
    "SHUTDOWN ALIASES": [
        "shut down", "shutdown", "go offline", "power down", "sign off",
        "turn off jarvis", "turn jarvis off", "turn yourself off",
        "turn off yourself",
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
    # 2026-09-06: 'hand mouse off' → air_control_off is this section's own
    # example, and it loaded AIR-MOUSE alone — which documents air_mouse_disarm
    # for that same phrase. The prompt deliberately gives BOTH sections the
    # 'hand mouse on/off' and 'give me the cursor' triggers and then tells the
    # model they are separate subsystems ("separate from the air-mouse above").
    # That disambiguation is only teachable when both bodies are in front of it,
    # so these keywords overlap AIR-MOUSE's on purpose — do not "resolve" the
    # ambiguity by giving the phrase to one section; the prompt owns that call.
    "AIR CONTROL": [
        "air control", "spatial mouse", "reach out", "grab and drag",
        "fist grab", "kinect mouse", "movie-style",
        "hand mouse", "control the mouse", "give me back the mouse",
        "give me the cursor",
    ],
    "MUSIC + VIDEO PLAYBACK": [
        "playback", "play a video", "media keys", "play/pause", "media control",
        "resume playback", "pause playback",
    ],
    "STREAMING SERVICES": [
        "netflix", "hulu", "disney", "disney+", "hbo", "prime video",
        "streaming service", "watch on", "auto-play", "put on a movie",
        # 2026-09-06: the Apple Music APP-LIFECYCLE actions live at the foot of
        # this section, and its own examples for them routed to MUSIC CONTROLS
        # instead — where the only nearby tokens are pause_music/stop. That is
        # an actively harmful near-miss, not a silent one: 'stop keeping Apple
        # Music open' would have STOPPED THE MUSIC rather than cancelled the
        # keep-alive. NOTE "keep apple music" cannot match "keepING apple
        # music", so the gerund needs its own entry.
        "keep apple music", "stop keeping", "keep music", "keep it open",
        "keep it running", "always open apple music", "in the tray",
    ],
    "TASTE-AWARE MUSIC": [
        "my music taste", "recommend a song", "recommend music", "based on my taste",
        "music recommendation", "something i'd like", "music i'd like",
        # 2026-09-06: 'what have I been listening to lately?' → music_history
        # is this section's own example and it selected MUSIC CONTROLS (on the
        # keyword "listen") + STATUS READ-BACKS, neither of which carries
        # music_history. The taste keywords above are all REQUEST-shaped
        # ("recommend…"); every read-back phrasing was missing.
        "been listening", "listening history", "listen history",
        "recently played", "been playing lately", "last few songs",
    ],
    "FOCUS MODE / DO-NOT-DISTURB": [
        "focus mode", "do not disturb", "hold my notifications", "heads down",
        "what did i miss", "recap what i missed",
        # documented trigger that used to route only on the header word "mode"
        "quiet mode",
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
        # 2026-09-29: the section's own future-day example ('will it rain
        # tomorrow') loaded only EVENING / DAILY BRIEFING via "tomorrow".
        "will it rain", "going to rain", "chance of rain",
        # 2026-10-01: "what's it like outside" named no weather word, so its
        # follow-up 'and what about tomorrow?' had nothing to inherit.
        "like outside",
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
        # 2026-09-29: the trigger phrases this body prints for status_panel and
        # system_pulse. 'JARVIS, system status' -> status_panel used to reach
        # the model only because the 6.7k-char STATUS READ-BACKS section loaded
        # on the bare header word "status" and happens to cross-reference
        # status_panel; with "status" generic, route the home section instead.
        "system status", "status report", "quick status", "pulse check",
        "give me a pulse", "how are the systems", "system readout",
        "bring up the diagnostics", "full diagnostic", "everything looking",
    ],
    "MULTI-STEP TASKS": [
        "add to cart", "find and add", "and add to", "buy me", "order online",
        "checkout", "multi-step", "then click", "do all of",
    ],
    "LOCAL VOICE CLONE": [
        "voice clone", "cloned voice", "voice profile", "clone voice",
        "your own voice", "in-character voice", "chatterbox", "switch to my voice",
        # 2026-09-06: 'stop cloning' → disable_voice_clone selected nothing.
        # Every spelling here was a NOUN ("voice clone", "cloned voice"); the
        # gerund the body itself prints as the trigger had no entry. Bare
        # "cloning" is safe — the word appears nowhere else in this grammar.
        "cloning", "normal voice", "what voices",
    ],
    "SMART HOME — PER-BRAND LIST": [
        "smart device",
        "list my lights", "list plugs", "which lights", "hue list", "govee list",
        "kasa list", "per brand", "brand list", "list smart",
        # documented triggers that used to route only on the header word
        # "list" (now generic -- " list" also matched " listener")
        "list my hue", "list my tuya", "list my govee", "list my kasa",
        "list my lifx", "list my plugs",
    ],
    "WAKE-WORD MODE": [
        "wake word mode", "wake-word mode", "require my name", "require your name",
        "always listening", "manual wake", "gate on wake",
        # documented triggers: 'music mode' routed only on the header word
        # "mode"; the rest never routed here at all.
        "music mode", "require the wake word", "only answer when i say",
        "normal listening", "always listen",
    ],
    "AMBIENT-LEARNING MODE": [
        "ambient learning", "listen and learn", "go quiet", "keep learning",
        "stay talkative", "answer then go quiet",
    ],
    # The double-clap trigger (skills/clap_trigger.py). Multi-word keywords
    # only: a bare "clap" would load this for "play Eric Clapton" (keywords
    # match at a word start and may run on).
    "CLAP TRIGGER": [
        "clap trigger", "clap detection", "clap detector", "double clap",
        "double-clap", "clap twice", "clap to wake", "clap routine",
        "clapping trigger", "the clapper",
    ],
    # ── Sections added 2026-09-04 with the reachability work. Documenting an
    # action in core/prompts.py is only HALF of making it reachable: a section
    # the router never selects reaches the model as a name in the capability
    # INDEX and nothing else, so the model sees that the capability exists but
    # never its instructions or trigger phrases. tests/test_prompt_router.py
    # fails when a recognised section has no routing, which is what caught
    # these — keep that test green rather than adding a section here silently.
    # Verified live 2026-09-04: without these, "what microphone are you using"
    # selected only MULTI-MONITOR APP LAUNCHING / PHONE NOTIFICATIONS / PHONE
    # BRIDGE (the last two by the substring "phone" inside "microphone"), so
    # what_microphone never reached the model while system_pulse did — which is
    # precisely the mis-route the owner reported.
    "AUDIO DEVICES — WHICH MICROPHONE AND SPEAKERS ARE IN USE": [
        "microphone", "mic", "what mic", "which mic", "listening on",
        "hearing me", "heard on", "heard me", "being heard", "picking me up",
        "speaker", "speakers", "headphones", "earphone", "playing through",
        "speaking through", "audio device", "audio devices", "input device",
        "output device", "sound device", "what are you using to hear",
        "what am i speaking into",
    ],
    "EVENING / DAILY BRIEFING": [
        "evening briefing", "daily briefing", "brief me", "end of day",
        "before bed", "tomorrow", "what's tomorrow", "run the briefing",
        "night briefing", "recap",
    ],
    "MEMORY MAINTENANCE": [
        "back up your memory", "export your memory", "snapshot your memory",
        "save a copy of what you know", "forget the last", "forget last hour",
        "forget that", "wipe your memory", "clear your memory",
        "what do you remember", "recent facts", "memory file",
        # 2026-09-06: 'JARVIS, forget everything you know about me' →
        # reset_memory selected SESSION MEMORY RECALL (on "forget"), whose only
        # nearby token is the PARTIAL forget. The scope note in this very body
        # — "Only an explicit 'everything' / 'all of it' / 'start over' earns
        # reset_memory" — was therefore never in front of the model on the one
        # turn it exists to govern. A near-miss here is a DATA outcome, not a
        # wording one: the owner asks for a full wipe and gets an hour dropped.
        "forget everything", "erase your memory", "forget all of it",
        "start over and forget", "scrub the last hour",
        "never happened",
        # 2026-10-01 brain eval (mem-03): 'scrub everything from the past
        # hour' loaded only TIMERS (on "hour"), so forget_last_hour never
        # reached the model. "past hour" was spelled nowhere; every entry is
        # a removal verb or the 'everything from the … hour' shape.
        "scrub everything", "scrub the past hour", "forget the past hour",
        "wipe the last hour", "wipe the past hour", "erase the last hour",
        "erase the past hour", "clear the last hour", "clear the past hour",
        "delete the last hour", "delete the past hour",
        "everything from the last hour", "everything from the past hour",
    ],
    "PENDING PROMISES": [
        "promise", "promises", "waiting on", "still pending", "owe me",
        "going to tell me", "let me know when", "outstanding",
    ],
    "SELF-TEST PROBES": [
        "test the mic", "test my mic", "test microphone", "mic test",
        "test your speech", "test tts", "test the camera", "test vision",
        "test your voice", "self test", "self-test", "probe", "is my mic working",
        # 2026-09-06: every keyword above starts "test …", so the two examples
        # the body prints in any OTHER shape both missed.
        #   'check the webcam' -> WEBCAM AWARENESS + UNIFIED, i.e. the model was
        #     shown the camera-LOOK actions and asked to run a camera PROBE.
        #     The body's own "'test the camera' / 'check the webcam'" pairing is
        #     what makes both spellings this section's, so route both.
        #   'how fast is your brain right now' -> LOCAL MODEL SELECTION (on
        #     "your brain"), which names models but never latency_benchmark.
        # "why are you slow" is deliberately NOT added: it already routes to
        # UNREADABLE STATE, whose body cites [ACTION: latency_benchmark] as the
        # one measurable thing — so that phrasing is answered, and duplicating
        # it here would put two competing sections on the same turn.
        "check the webcam", "check the camera", "test the webcam",
        "test all your skills", "test your skills",
        "how fast is your brain", "how fast are you", "latency",
    ],
    # The unreadable-state section is the one that teaches an HONEST refusal —
    # it exists precisely so a question with no handler stops coming back as a
    # confidently wrong neighbouring action (the 2026-09-04 "what microphone are
    # you using" -> system_pulse CPU read-out). Its keywords must therefore
    # cover the QUESTIONS, not any action name, because there is no action.
    "MUTE / DEAF / SLOW / WHISPER — UNREADABLE STATE": [
        "echo cancellation", "noise suppression", "agc", "audio processing",
        "are you muted", "am i muted", "is your mic muted", "muted",
        "whisper model", "what model are you using for speech", "cuda or cpu",
        "why are you slow", "are you deaf", "can you hear me",
    ],
    # STATUS READ-BACKS — the "is it ON?" block (20 <feature>_status
    # actions). Until 2026-09-05 its opening line was PROSE, not a header,
    # so split_pc_control folded the whole block into SUIT DIAGNOSTICS and
    # all 17 trigger phrases it documents missed it. Now that the header
    # parses, these keywords are the other half: the user never says the
    # action name, he asks a QUESTION ("is the workshop HUD showing", "are
    # you listening in the background"), so route on the questions. A miss
    # here is not a quiet degrade — PC_CONTROL_SAFETY_RULES ships its
    # closed-list rule on every turn, so an unreached read-back comes back
    # as "I've no way to check that, sir" for a capability that exists.
    "STATUS READ-BACKS": [
        # generic "is it on / still running" shapes
        "is it on", "is it still", "is it running", "is that on",
        "still on", "still up", "still running", "still going",
        "still active", "are you still", "are you running",
        # on-screen surfaces
        "holographic", "holo overlay", "workshop hud", "workshop mode",
        "printer camera", "chamber camera", "bambu camera", "arc reactor",
        "stark ring", "status ring", "is the hud", "hud up", "hud showing",
        # background watchers
        "listening", "watching my screen", "are you watching", "ambient",
        "extractor", "learned anything", "keeping an eye", "anticipation",
        "predicting", "briefing me", "weekly digest", "banter",
        "making jokes", "robot build", "smart home router", "not respond",
        "didn't respond", "outbound gate", "gate armed", "draft preview",
    ],
}

# Header words that must NOT pull a section in on their own. select_sections
# falls back to a section's header words when no curated keyword fires, and the
# fallback is a loose prefix/suffix test (" mode" also matches " model"). These
# words sit in many headers without saying WHICH capability is meant, so one of
# them in a turn loaded every section that carried it. Measured on the live
# prompt 2026-09-29 before this set existed:
#   'turn off wake word mode'  -> 9 sections / ~8.5k chars of volatile tail:
#                                 GUARD, FOCUS x2, NIGHT-OWL and AMBIENT-LEARNING
#                                 MODE all loaded on the one word "mode".
#   'Jarvis now has full control over the robot'
#                              -> AIR CONTROL + POINT-TO-CONTROL on "control",
#                                 i.e. Kinect mouse / pointing grammar handed to
#                                 the model for a sentence about a robot.
#   'which model are you using' -> every *MODE section (" mode" in " model").
# Only the HEADER-WORD fallback ignores these. An explicit _SECTION_KEYWORDS
# entry is curated and still matches ("system" still routes SYSTEM HEALTH, "wake
# word mode" still routes WAKE-WORD MODE), so a section that needs one of these
# words gets it as a keyword phrase, never as a bare header word.
_GENERIC_HEADER_WORDS = frozenset({
    "mode", "control", "project", "status", "system", "list", "check",
    # Plural of the above; only MUSIC CONTROLS carries it, and "music" is its
    # real header trigger.
    "controls",
    # Not generic English, but ambiguous in THIS prompt: it heads both WAKE
    # LISTENER (the porcupine hotword engine) and WAKE-WORD MODE (require a
    # leading 'JARVIS'), and "wake me up at seven" is a TIMERS turn. Routing
    # on it alone put wake_listener_stop next to wake_word_mode_off for
    # 'turn off wake word mode'; the two keyword lists tell them apart.
    "wake",
    # 2026-10-01: "how smart are you" loaded SMART HOME, SMART HOME DISCOVERY
    # and the PER-BRAND LIST on "smart" alone — lights grammar for a question
    # about intelligence. Their keywords ("smart home", "list smart", ...)
    # and the header word "home" still route every real smart-home turn.
    "smart",
    # Heads SELF-PRESERVATION, SELF DIAGNOSTIC, SELF-TEST PROBES and
    # SELF-KNOWLEDGE, so "run a self test" (or "take a selfie": the match is
    # word-START) loaded all four. Each has its own keywords or other header
    # words ("preservation", "diagnostic", "self test").
    "self",
    # Heads only SELF-KNOWLEDGE, whose keywords and predicate route it:
    # "to my knowledge it's fine" loaded it (review 2026-10-02).
    "knowledge",
    # CLAP TRIGGER's header words: the word-start match took "play Eric
    # Clapton" and "clap along" ("clap") and "trigger the alarm" ("trigger")
    # into the clap-trigger grammar. Its keyword phrases route it instead.
    "clap", "trigger",
})

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


# Unit vocabulary that routes hardware / weather sections on an ordinary turn
# ("how hot is the GPU in celsius") but is pure arithmetic on a unit-conversion
# turn. 2026-09-29 live: "convert 100 degrees fahrenheit to celsius" loaded
# SYSTEM HEALTH (gpu_usage / temperature grammar) on "degrees" + "celsius". On
# a conversion turn these keywords do not count; any OTHER keyword still does
# ("convert the gpu temperature to fahrenheit" keeps SYSTEM HEALTH via "gpu").
_CONVERSION_NEUTRAL_KEYWORDS = frozenset({
    "degrees", "celsius", "fahrenheit", "kelvin", "temperature", "temp",
})


# SELF-KNOWLEDGE by shape (review 2026-10-02): an AI brand or a comparison
# word alone is not a question about JARVIS ("use Claude", "compared to last
# week"); "you" + a comparison + an AI / model name is.
_SK_WAKE_LEAD_RE = re.compile(r"^\W*(?:(?:hey|ok|okay)\W+)?jarvis\b\W*")
_SK_YOU_RE = re.compile(r"\b(?:you|you're|youre|your|yourself|jarvis)\b")
_SK_COMPARE_RE = re.compile(
    r"\b(?:compar\w*|stacks?\s+up|versus|vs|smarter|dumber|better|worse|"
    r"faster|slower|stronger|weaker|more\s+capable|as\s+good\s+as)\b")
_SK_OTHER_AI_RE = re.compile(
    r"\b(?:claude|opus|sonnet|haiku|gpt\w*|chatgpt|gemini|llama|gemma|qwen|"
    r"mistral|copilot|siri|alexa|grok|other\s+ais?|other\s+assistants?|"
    r"models?|ai\s+assistants?|llms?)\b")
_SK_HOW_GOOD_RE = re.compile(
    r"\bhow\s+(?:good|smart|intelligent|clever|capable|bright|advanced|"
    r"powerful)\s+(?:are\s+you|is\s+jarvis)\b(?!\s+at\b)")


def is_self_knowledge_request(user_text: str) -> bool:
    """A question about JARVIS's own capability against other AIs: "how good
    are you" (not "how good are you at chess"), or "you" + a comparison + an
    AI name / "other assistants" ("are you smarter than ChatGPT", "how do you
    stack up against GPT"). A leading wake word is not the "you". Never
    raises."""
    try:
        low = _SK_WAKE_LEAD_RE.sub("", str(user_text or "").lower())
        if _SK_HOW_GOOD_RE.search(low):
            return True
        return bool(_SK_YOU_RE.search(low) and _SK_COMPARE_RE.search(low)
                    and _SK_OTHER_AI_RE.search(low))
    except Exception:
        return False


# Sections a turn implicates by SHAPE rather than by a word (2026-10-01). Spoken
# arithmetic is the case: "what's 12 times 7" named no PYTHON SANDBOX keyword,
# so run_python (the calculator) never reached the local model, and an
# operator word cannot be a keyword on its own ("what times does the store
# open"). core.spoken_math.is_arithmetic_request wants a number on BOTH sides.
# SELF-KNOWLEDGE: is_self_knowledge_request above.
_PREDICATE_ROUTES = {
    "PYTHON SANDBOX": is_arithmetic_request,
    "SELF-KNOWLEDGE": is_self_knowledge_request,
}

# Short keywords ("tv", "ram", "hot", "mic", "bed", "obs", "lan") may only take
# a plural after them; longer ones may run on freely (see _keyword_hit).
_SHORT_KEYWORD_LEN = 3
_SHORT_KEYWORD_TAILS = ("", "s", "es")


def _keyword_hit(kw: str, low: str) -> bool:
    """True when ``kw`` occurs in ``low`` (the space-padded, lower-cased turn)
    starting at a WORD BOUNDARY.

    The router used to test ``kw in low`` — a bare substring — so a keyword
    that began mid-word routed a section the turn never named (2026-10-01):
    "phone" inside "microphone" loaded PHONE NOTIFICATIONS + PHONE BRIDGE on
    every microphone question, "face" inside "interface" FACE RECOGNITION on
    every web-interface turn, "hot" inside "hotword" / "screenshot" SYSTEM
    HEALTH, "lan" inside "plans" the network section, "hbo" inside
    "dashboard" STREAMING SERVICES. A keyword now has to START a word (or
    itself start with a non-alphanumeric, like the deliberate " c drive").

    The END is still open, because many keywords are stems by design ("print"
    -> "printing", "remind" -> "reminders", "throttl", "auto switch" ->
    "auto switching"): except for a SHORT keyword (<= _SHORT_KEYWORD_LEN
    alphanumerics), which may only take a plural ("tv" -> "tvs", "ram" ->
    "ram's") — never "hot" -> "hotword", "mic" -> "michael", "ram" ->
    "random", "bed" -> "bedroom". Multi-word phrases follow the same rule at
    their first and last word. Never raises."""
    if not kw or not low:
        return False
    n = len(kw)
    short = n <= _SHORT_KEYWORD_LEN and kw.isalnum()
    start = 0
    while True:
        i = low.find(kw, start)
        if i < 0:
            return False
        start = i + 1
        if kw[0].isalnum() and i > 0 and low[i - 1].isalnum():
            continue                        # begins mid-word
        if short:
            j = k = i + n
            while k < len(low) and low[k].isalnum():
                k += 1
            if low[j:k] not in _SHORT_KEYWORD_TAILS:
                continue                    # "hot" -> "hotword"
        return True


# ── Follow-up routing (2026-10-01 brain eval) ────────────────────────────
# A follow-up names its subject only in an EARLIER turn. 'Never mind, cancel
# that.' after 'set a timer for ten minutes' routed nothing, so cancel_timer
# never reached the model; 'Pause it.' after "how's the print going?" routed
# MUSIC CONTROLS alone, so pause_print was invisible and pause_music the only
# pause on offer. When THIS turn is short and elliptical, the router also
# reads the previous user turn (and one more if that one was elliptical too).
# Words alone still route as before; history can only ADD sections, and only
# on a short elliptical turn, so an ordinary turn's prompt does not grow.
_FOLLOWUP_MAX_WORDS = 6
_FOLLOWUP_MAX_HOPS = 2
# Not counted toward the word limit (wake word, fillers, acknowledgements).
_FOLLOWUP_FILLERS = frozenset({
    "jarvis", "hey", "ok", "okay", "please", "sir", "um", "uh", "oh", "ha",
    "hm", "hmm", "well", "so", "right",
})
# A word that always points back at something the turn does not name.
# ("it" and "that's" are judged by position: _followup_kind.)
_FOLLOWUP_REFERENTS = frozenset({
    "them", "they", "other", "same", "again",
})
# Demonstratives point back only when they stand ALONE ('cancel that', 'is
# that normal', 'skip this one'). Before a noun they are determiners and the
# turn names its own subject: 'this afternoon', 'that timer'.
_FOLLOWUP_DEMONSTRATIVES = frozenset({"this", "that", "these", "those"})
_FOLLOWUP_DEMONSTRATIVE_NEXT = frozenset({
    "one", "ones", "again", "too", "instead", "now", "for", "to", "on", "off",
    "up", "down", "in", "into", "out", "over", "back",
})
_FOLLOWUP_COPULAS = frozenset({"is", "was", "are", "were", "isn't", "wasn't"})
# Openers that continue the previous request ('what about tomorrow').
_FOLLOWUP_LEADS = ("what about", "how about", "never mind", "nevermind")
# "and ..." / "also ..." continue it only as a FRAGMENT ("and tomorrow?",
# "and the bedroom"): "and play some jazz" is a request of its own (review
# 2026-10-02). Up to this many words after the conjunction is a fragment;
# a longer one inherits only when its own words route nothing.
_FOLLOWUP_CONJUNCTIONS = ("and", "also")
_FOLLOWUP_FRAGMENT_MAX_WORDS = 2
# Review 2026-10-02: "it" in subject position is often EXPLETIVE - "how's it
# going", "what's it like outside", "is it going to rain", "it's cold" -
# and pointed back at nothing, yet routed the previous turn's sections in. In
# those positions "it" counts only before a word that describes a THING's
# state ("is it on", "is it done yet", "it's too loud").
_EXPLETIVE_IT_BEFORE = frozenset({
    "is", "was", "will", "would", "isn't", "wasn't", "what's", "whats",
    "how's", "hows", "where's", "when's",
})
_IT_STATE_WORDS = frozenset({
    "on", "off", "done", "finished", "working", "running", "playing", "open",
    "closed", "ready", "over", "still", "loud", "quiet", "bright", "dim",
    "broken", "back", "up", "down", "charged", "connected", "paused",
    "stopped", "printing", "recording", "plugged", "out", "too", "armed",
    "muted", "loading", "frozen", "stuck", "set", "going", "right", "wrong",
    "doing", "saying", "showing", "reading",
})
# "going" above is "is it going" (a print, a download); the weather and
# small-talk forms are excluded by the word after it:
_IT_GOING_EXPLETIVE_NEXT = frozenset({"to", "gonna", ""})
# "that's" + one of these is an acknowledgement ("that's great, thanks",
# "that's all"), not a pointer at a thing.
_THATS_ACK = frozenset({
    "", "great", "good", "fine", "perfect", "cool", "awesome", "nice",
    "right", "correct", "true", "enough", "all", "it", "okay", "ok",
    "alright", "amazing", "wonderful", "brilliant", "excellent",
    "interesting", "funny", "hilarious", "fair", "lovely", "helpful",
    "impressive", "incredible", "fantastic", "neat", "sweet", "kind",
    "weird", "crazy", "wild", "sad", "unfortunate", "terrible", "awful",
    "what", "how", "why", "life", "fun",
})


def _it_points_back(words: List[str], i: int) -> bool:
    """Whether ``words[i]`` ("it" / "it's") points back at a thing."""
    w = words[i]
    prev = words[i - 1] if i > 0 else ""
    nxt = words[i + 1] if i + 1 < len(words) else ""
    nxt2 = words[i + 2] if i + 2 < len(words) else ""
    if w in ("it's", "its"):
        state, after = nxt, nxt2
    elif prev in _EXPLETIVE_IT_BEFORE:
        state, after = nxt, nxt2
    elif nxt in ("is", "was", "will"):
        state, after = nxt2, (words[i + 3] if i + 3 < len(words) else "")
    else:
        return True                  # "turn it off", "pause it", "make it 15"
    if state == "going":
        return after not in _IT_GOING_EXPLETIVE_NEXT and prev not in (
            "how's", "hows")
    return state in _IT_STATE_WORDS


def _followup_kind(user_text: str) -> str:
    """Why ``user_text`` leans on an earlier turn: "referent" (a word that
    points back), "lead" (a continuing opener or a short "and ..."
    fragment), "conjunction" (a longer "and ..." / "also ..."), or "" (it
    does not). Never raises."""
    try:
        words = [w for w in re.findall(r"[a-z0-9']+", (user_text or "").lower())
                 if w not in _FOLLOWUP_FILLERS]
    except Exception:
        return ""
    if not words or len(words) > _FOLLOWUP_MAX_WORDS:
        return ""
    for i, w in enumerate(words):
        if w in ("it", "it's", "its"):
            if w != "its" or i == 0:
                if _it_points_back(words, i):
                    return "referent"
            continue
        if w == "that's":
            if (words[i + 1] if i + 1 < len(words) else "") not in _THATS_ACK:
                return "referent"
            continue
        if w in _FOLLOWUP_REFERENTS:
            return "referent"
        if w in _FOLLOWUP_DEMONSTRATIVES and (
                i == len(words) - 1
                or words[i + 1] in _FOLLOWUP_DEMONSTRATIVE_NEXT
                or (i > 0 and words[i - 1] in _FOLLOWUP_COPULAS)):
            return "referent"
    rest = words
    conj = False
    while rest and rest[0] in _FOLLOWUP_CONJUNCTIONS:
        rest = rest[1:]
        conj = True
    joined = " ".join(rest)
    if any(joined == lead or joined.startswith(lead + " ")
           for lead in _FOLLOWUP_LEADS):
        return "lead"
    if conj:
        return ("lead" if len(rest) <= _FOLLOWUP_FRAGMENT_MAX_WORDS
                else "conjunction")
    return ""


def is_elliptical_followup(user_text: str) -> bool:
    """True when ``user_text`` is a short turn that leans on an earlier one:
    at most _FOLLOWUP_MAX_WORDS words (fillers and the wake word not
    counted) AND either a back-reference ('it' as an object or with a
    thing's state - 'turn it off', 'is it done yet' - 'them', 'cancel that',
    'the other one'; not the expletive 'it' of "how's it going" or the
    acknowledgement "that's great") or a continuing opener ('what about …',
    'never mind', 'and …'). Never raises."""
    return bool(_followup_kind(user_text))


def _message_text(content) -> str:
    """Plain text of a chat message's content (a string, or a list of
    content blocks as the cloud API takes them)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def _prior_user_turns(history, current: str) -> List[str]:
    """The user turns before ``current``, newest first. ``history`` may end
    with ``current`` itself (live: _call_llm appends the turn before building
    the prompt) or stop just before it (the eval harness); both work."""
    users = []
    for msg in history or ():
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _message_text(msg.get("content")).strip()
            if text:
                users.append(text)
    if users and users[-1] == (current or "").strip():
        users.pop()
    return users[::-1]


def _routing_parts(user_text: str, history=None) -> List[str]:
    """``[user_text]``, or — for a short elliptical follow-up with history —
    ``[user_text, previous user turn(, the one before)]``. Never raises."""
    text = user_text or ""
    try:
        if not history or not is_elliptical_followup(text):
            return [text]
        parts = [text]
        for prev in _prior_user_turns(history, text)[:_FOLLOWUP_MAX_HOPS]:
            parts.append(" ".join(prev.split()))
            if not is_elliptical_followup(prev):
                break
        return parts
    except Exception:
        return [text]


def routing_text(user_text: str, history=None) -> str:
    """The text the router matches keywords against: ``user_text`` alone,
    or — for a short elliptical follow-up — ``user_text`` plus the previous
    user turn(s), one per line. Lines never fuse: no keyword spans a newline,
    so "cancel that" + "timer ..." cannot form a phrase neither turn said."""
    return "\n".join(_routing_parts(user_text, history))


# A section that documents a self-terminating action (core.action_risk:
# shutdown_jarvis and its aliases, restart, upgrade, start_overnight_upgrade)
# names an action that runs at once with no confirmation. Review 2026-10-02:
# 'Okay, turn it off.' after "why did you shut down earlier?" inherited
# SHUTDOWN ALIASES, putting the exact name shutdown_jarvis in front of the
# model in the 10-01 incident's own setting. Such a section routes only on
# the current turn's OWN words, never through history. "Documents" means the
# name as an identifier (shutdown_jarvis), as an action token ([ACTION:
# restart]) or as a list entry ("  restart      — relaunch ..."); prose that
# merely says "restart" ("persists across restart") is not.
_SELF_TERM_UNDERSCORED = sorted(n for n in SELF_TERMINATING_ACTIONS if "_" in n)
_SELF_TERM_BARE = sorted(n for n in SELF_TERMINATING_ACTIONS if "_" not in n)
_SELF_TERM_DOC_RE = re.compile(
    r"(?<![a-z0-9_])(?:" + "|".join(map(re.escape, _SELF_TERM_UNDERSCORED))
    + r")(?![a-z0-9_])"
    + r"|\[ACTION:\s*(?:" + "|".join(map(re.escape, sorted(SELF_TERMINATING_ACTIONS)))
    + r")\b"
    + r"|^[ \t]+(?:" + "|".join(map(re.escape, _SELF_TERM_BARE))
    + r")[ \t]{2,}", re.MULTILINE)


def documents_self_terminating_action(body: str) -> bool:
    """True when a section ``body`` documents a self-terminating action (see
    _SELF_TERM_DOC_RE). Never raises; a fault counts as True (the section
    then just routes on the turn's own words)."""
    try:
        return bool(_SELF_TERM_DOC_RE.search(body or ""))
    except Exception:
        return True


def _match_sections(parts: List[str], sections: List[Tuple[str, str]]) -> set:
    """Names of the sections ``parts`` (one or more turns' text, matched as
    separate lines) route to, the always-on set included."""
    user_text = "\n".join(parts)
    low = " " + (user_text or "").lower() + " "
    neutral = (_CONVERSION_NEUTRAL_KEYWORDS
               if any(is_unit_conversion_request(p) for p in parts)
               else frozenset())
    hits = set()
    for header, _body in sections:
        name = header.strip()
        upper = name.upper()
        hit = name.upper() in _ALWAYS
        if not hit:
            # header words present in the query? (generic ones never count
            # on their own -- see _GENERIC_HEADER_WORDS). Word-start matched
            # like the keywords: the old " w" / "w " test also took a match
            # at the END of a longer word ("yourself" loaded every SELF-*
            # section, "tonight" NIGHT-OWL MODE, "unread" STATUS READ-BACKS).
            words = [w for w in re.split(r"[^a-z0-9]+", name.lower())
                     if len(w) > 3 and w not in _GENERIC_HEADER_WORDS]
            if any(_keyword_hit(w, low) for w in words):
                hit = True
        if not hit:
            for kw in _keywords_for(name):
                if kw not in neutral and _keyword_hit(kw, low):
                    hit = True
                    break
        if not hit:
            pred = _PREDICATE_ROUTES.get(upper)
            if pred is not None:
                try:
                    hit = any(bool(pred(p)) for p in parts)
                except Exception:
                    hit = False
        if hit:
            hits.add(name)
    return hits


def _select(user_text: str, sections: List[Tuple[str, str]], history=None):
    """``(included, dropped, inherited)`` - see select_sections. ``inherited``
    is the set of included sections only the history routed."""
    own = _match_sections([user_text or ""], sections)
    inherited: set = set()
    parts = _routing_parts(user_text, history)
    if len(parts) > 1 and _followup_kind(user_text) == "conjunction" and (
            own - {n for n in own if n.upper() in _ALWAYS}):
        # "and play some jazz": a request of its own, which routes itself.
        parts = parts[:1]
    if len(parts) > 1:
        bodies = {h.strip(): b for h, b in sections}
        for name in _match_sections(parts, sections) - own:
            if not documents_self_terminating_action(bodies.get(name, "")):
                inherited.add(name)
    keep = own | inherited
    included: List[str] = []
    dropped: List[str] = []
    for header, _body in sections:
        name = header.strip()
        (included if name in keep else dropped).append(name)
    return included, dropped, inherited


def select_sections(user_text: str, sections: List[Tuple[str, str]],
                    history=None) -> Tuple[List[str], List[str]]:
    """Return (included_section_names, dropped_section_names) for `user_text`.

    ``history`` (optional, chat messages) lets a short elliptical follow-up
    route on the previous user turn as well - see routing_text - but never
    to a section that documents a self-terminating action (those route on
    the turn's own words only), and not for a longer "and ..." turn whose
    own words already route something. Without it, or on any other turn,
    routing is exactly the words of ``user_text``."""
    included, dropped, _inherited = _select(user_text, sections, history)
    return included, dropped


# Sections whose body is RENDERED from live runtime values each time it ships,
# instead of PC_CONTROL_PROMPT's static text (2026-10-01). SELF-KNOWLEDGE names
# the active local model tag, which set_model changes at runtime — a tag frozen
# into the prompt would be the stale-duplicate bug class. The static copy in
# PC_CONTROL_PROMPT still parses, routes and indexes like any other section;
# only the text handed to the model is swapped. Rendering happens in the
# per-turn halves (slim_pc_control / turn_pc_block), never in stable_pc_block,
# so the KV-cached prefix stays byte-identical whatever model is loaded.
def _render_self_knowledge() -> str:
    from core.prompts import render_self_knowledge_section
    return render_self_knowledge_section()


_RUNTIME_SECTIONS = {
    "SELF-KNOWLEDGE": _render_self_knowledge,
}


def _section_text(header: str, body: str) -> str:
    """The text to ship for a selected section: its live render when it has
    one (_RUNTIME_SECTIONS), else ``body``. Never raises — a failed render
    ships the static body."""
    render = _RUNTIME_SECTIONS.get(header.strip().upper())
    if render is None:
        return body
    try:
        out = render()
    except Exception:
        return body
    return out if isinstance(out, str) and out.strip() else body


def slim_pc_control(user_text: str, pc_control: str, history=None) -> str:
    """Build a slimmed PC_CONTROL for this turn: core preamble + the sections the
    text implicates + a one-line INDEX of what was left out (so the model still
    knows those capabilities exist). Falls back to the full text if parsing finds
    no sections. ``history``: see select_sections. Never raises — a bad parse
    returns the full prompt."""
    try:
        core, sections = split_pc_control(pc_control)
        if not sections:
            return pc_control
        included, dropped = select_sections(user_text, sections, history)
        inc_set = set(included)
        parts = [core]
        for header, body in sections:
            if header.strip() in inc_set:
                parts.append(_section_text(header, body))
        if dropped:
            parts.append(
                "\n\nADDITIONAL CAPABILITIES (ask and I'll use them; full "
                "instructions load on request): " + "; ".join(dropped) + ".")
        return "\n".join(parts)
    except Exception:
        return pc_control


# ──────────────────────────────────────────────────────────────────────────
#  CACHE-STABLE SPLIT  (2026-09-06 latency work)
# ──────────────────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS — and why slim_pc_control above is not enough.
#
# slim_pc_control does its job: it cuts PC_CONTROL from ~126k chars to ~6k and
# costs 1.4 ms. But its output CHANGES every turn, and it is spliced into the
# MIDDLE of the system prompt (~46 % depth). That single fact was, measured,
# 40 % of JARVIS's entire speak-to-speak latency.
#
# The local brain (gemma4:26b-a4b-it-qat) is a SLIDING-WINDOW-attention model:
# llama.cpp keeps a full KV cache for only 5 of its 30 layers and a 2048-cell
# SWA cache for the other 25. Reusing a prefix therefore needs a "context
# checkpoint" — a snapshot of that 2048-cell SWA state — and llama.cpp can only
# restore one when the divergence point is inside it. Measured on this box with
# a production-sized 11.4k-token prompt (scratchpad e1_distance.py, n=3 each):
#
#     divergence   53 tokens from the end of the previous prompt →    68 ms
#     divergence  535 tokens from the end                        →   168 ms
#     divergence 1070 tokens from the end                        →  2506 ms
#     divergence 2406 tokens from the end                        →  2534 ms
#     divergence 10695 tokens from the end                       →  2536 ms
#     pure append (identical prompt, new user text)              →    30 ms
#
# There is no gentle degradation: past ~1024 tokens of divergence llama.cpp
# logs "forcing full prompt re-processing due to lack of cache data (likely due
# to SWA…)" and re-evaluates ALL 12.7k tokens at ~4,200 tok/s. And there is no
# cross-request rescue: A → B → A re-evaluates A in full every time (e2_cache.py),
# so keeping a handful of prompt variants in rotation does not help either.
#
# So the fix is not to make the volatile block smaller or to move it later in
# the system prompt — it is to get it OUT of the cached region entirely:
#
#   stable_pc_block()  →  byte-identical on every turn: the core preamble
#                         (action grammar + safety rules) plus an INDEX naming
#                         every capability. Lives where PC_CONTROL used to.
#   turn_pc_block()    →  just the section BODIES this turn implicates. The
#                         caller puts this at the head of the FINAL user
#                         message, i.e. after everything cached.
#
# Content-wise this is a SUPERSET of slim_pc_control's output (same core, same
# selected bodies, and an index of ALL sections rather than only the dropped
# ones), which is what tests/test_prompt_router.py::CacheStableSplitTests
# asserts utterance by utterance — the split may reorder what the model sees,
# never subtract from it.

_STABLE_BLOCK_CACHE: Dict[int, str] = {}

_INDEX_HEADER = (
    "\n\nCAPABILITY INDEX — every capability you have. The full instructions "
    "for the ones this turn needs are supplied with the user's message below; "
    "for anything else here, say you can do it and ask for the word:\n")


def stable_pc_block(pc_control: str) -> str:
    """The BYTE-STABLE half of the PC block: core preamble + an index naming
    every capability. Identical for every turn, so the KV prefix survives.

    Cached on the identity of ``pc_control`` — it is a module constant, so this
    is computed once per process. Never raises: a bad parse degrades to the
    core preamble alone, which still carries the action grammar and the safety
    rules, and turn_pc_block() still delivers the bodies."""
    key = id(pc_control)
    hit = _STABLE_BLOCK_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        core, sections = split_pc_control(pc_control)
        if not sections:
            out = pc_control
        else:
            names = [h.strip() for h, _b in sections]
            # The _ALWAYS sections are, by definition, in EVERY turn's
            # selection — so hoisting them up here changes nothing about what
            # the model sees and takes their bytes out of every volatile tail
            # forever. Free, and it is the only promotion that is provably
            # content-neutral.
            always = [b for h, b in sections if h.strip().upper() in _ALWAYS]
            out = ("\n".join([core] + always)
                   + _INDEX_HEADER + "; ".join(names) + ".")
    except Exception:
        out = pc_control
    _STABLE_BLOCK_CACHE[key] = out
    return out


def turn_pc_block(user_text: str, pc_control: str, history=None) -> str:
    """The VOLATILE half: the bodies of the sections `user_text` implicates.

    ``history`` (the chat so far) lets a short elliptical follow-up ('Pause
    it.', 'what about tomorrow') also route on the previous user turn — see
    routing_text. Returns '' when the turn implicates nothing beyond the
    always-on section set, so a turn that needs no extra instructions costs
    the cache nothing at all. Never raises — on a parse failure it returns
    the FULL section text, which is slower but never less informed than
    slim_pc_control was."""
    try:
        _core, sections = split_pc_control(pc_control)
        if not sections:
            return ""
        included, _dropped = select_sections(user_text, sections, history)
        inc = set(included)
        bodies = [_section_text(h, b) for h, b in sections
                  if h.strip() in inc and h.strip().upper() not in _ALWAYS]
        return "\n".join(bodies)
    except Exception:
        try:
            return "\n".join(b for _h, b in split_pc_control(pc_control)[1])
        except Exception:
            return pc_control


def inherited_turn_sections(user_text: str, pc_control: str,
                            history=None) -> set:
    """The section names turn_pc_block ships for ``user_text`` ONLY because
    of ``history`` (a short follow-up's inherited routing), always-on ones
    excluded. The local prompt budget ranks these below the turn's own
    long-term-memory recall (core.prompt_budget.RANK_INHERITED), so a
    spurious inheritance is the first thing an overflowing turn sheds.
    Never raises; set() on any fault."""
    if not history:
        return set()
    try:
        _core, sections = split_pc_control(pc_control)
        if not sections:
            return set()
        _inc, _drop, inherited = _select(user_text, sections, history)
        return {n for n in inherited if n.upper() not in _ALWAYS}
    except Exception:
        return set()


def split_turn_block(block: str) -> List[Tuple[str, str]]:
    """Split a turn_pc_block() string back into its sections, in order:
    ``[(header, text), ...]`` where ``"\\n".join(texts) == block`` exactly.

    The prompt budget (core/prompt_budget) drops whole sections from an
    overflowing turn, so it needs them one by one. Every body starts with
    its own header line and holds no other line that matches _HEADER_RE (a
    line that matches IS a header to split_pc_control), so splitting at
    those lines gets back exactly the selected sections. Text before the
    first header, which a real block never has, comes back under header ''.
    Never raises; on any fault the whole block is one part."""
    if not block:
        return []
    try:
        out: List[Tuple[str, str]] = []
        cur_head = ""
        cur: List[str] = []
        for ln in block.split("\n"):
            m = _HEADER_RE.match(ln.strip())
            if m and cur:
                out.append((cur_head, "\n".join(cur)))
                cur = []
            if m:
                cur_head = m.group("head").strip()
            cur.append(ln)
        out.append((cur_head, "\n".join(cur)))
        return out
    except Exception:
        return [("", block)]
