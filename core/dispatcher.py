"""command_chain_resolver — detect multi-step intents in a single utterance.

A user can say something like
  "play Michael Jackson and dim the lights and start a 45 minute focus timer"
and the LLM, given this as one ambiguous prompt, often drops one of the
intents or only emits a single [ACTION:] token. This module pre-resolves
the utterance: if it cleanly splits into 2+ recognized commands, each is
dispatched directly via the actions dict and a single consolidated
confirmation is returned for TTS. Anything that doesn't cleanly resolve
falls through to the LLM (the resolver returns None).

Public API:
  resolve_and_dispatch(utterance, actions) -> Optional[str]
      Returns a consolidated TTS confirmation if a chain was dispatched,
      else None (caller should fall through to the LLM as before).

  command_chain_resolver(utterance, available_actions) -> Optional[ChainResult]
      Pure resolution — splits and matches; does not execute. Useful for
      tests and for callers that want to inspect the plan before dispatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from core import streaming_search as _streaming_search
from core.failure_markers import FAILURE_MARKERS
from core.lead_fillers import strip_lead_filler as _strip_lead_filler


# ──────────────────────────────────────────────────────────────────────────
#  TYPES
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ChainStep:
    """One resolved intent within a chain."""
    action: str                    # action name in the ACTIONS dict
    arg: str                       # argument string for the action handler
    confirmation: str              # short phrase for the consolidated TTS line
    source: str                    # the raw segment from the utterance


@dataclass
class ChainResult:
    """Output of command_chain_resolver — the plan, not the execution."""
    steps: list[ChainStep] = field(default_factory=list)
    unknown: list[str]     = field(default_factory=list)   # unmatched segments


# ──────────────────────────────────────────────────────────────────────────
#  INTENT RULES
# ──────────────────────────────────────────────────────────────────────────
# Each rule binds one or more regex patterns to a target action name. The
# arg_fn turns the regex match into the string passed to the action.
# Rules are tried in order; first match wins. `requires_action` is the action
# name that must exist in the live ACTIONS dict for the rule to be eligible
# (defaults to the rule's `action`). Rules whose required action isn't loaded
# (skill not registered, etc.) are silently skipped.

_PUNCT_TAIL = re.compile(r"[.!?,;:]+\s*$")


def _strip(s: str) -> str:
    return _PUNCT_TAIL.sub("", s.strip())


def _arg_first_group(m: re.Match) -> str:
    g = m.group(1) if m.lastindex else ""
    return _strip(g) if g else ""


def _strip_play_filler(raw: str) -> str:
    """Drop a leading 'some ' / 'me some ' / 'me ' (original casing kept)."""
    raw_low = raw.lower()
    for filler in ("some ", "me some ", "me "):
        if raw_low.startswith(filler):
            raw = raw[len(filler):]
            break
    return raw.strip()


def _arg_play_music(m: re.Match) -> str:
    """Extract artist/song from 'play X' / 'put on X' / 'queue X'.

    If the user said 'play some <X>' or 'play me <X>', drop the filler so
    iTunes search gets the bare query.
    """
    # Preserve original casing where possible.
    return _strip_play_filler(_arg_first_group(m))


# A trailing "on <service>" names WHERE to play, so the request is not an
# Apple Music song search (2026-10-01). Without this rule the music rule took
# "play Stranger Things on Netflix" whole, and play_music resolved an iTunes
# SONG for "Stranger Things on Netflix" while the chain still said "music
# queued". Apple Music itself is deliberately absent: play_music already IS
# Apple Music.
_STREAMING_SERVICE_ALT = (
    r"netflix|you\s?tube|yt|spotify|hulu|disney(?:\s*\+|\s+plus)?"
    r"|(?:amazon\s+)?prime(?:\s+video)?|amazon|hbo(?:\s+max)?|max"
)


def _canon_streaming_service(svc: str) -> str:
    """Spoken service name → the _STREAMING_SERVICES key play_streaming takes.
    The video services resolve through the ONE verified table
    (core.streaming_search.canon_service, 2026-10-02); anything else
    (spotify) passes through as spoken."""
    s = re.sub(r"\s+", " ", (svc or "").strip().lower())
    return _streaming_search.canon_service(s) or s


def _arg_play_streaming(m: re.Match) -> str:
    """'play <title> on <service>' → the 'service|title' play_streaming takes."""
    title = _strip_play_filler(_strip(m.group(1) or ""))
    return f"{_canon_streaming_service(m.group(2) or '')}|{title}"


# ── Single-command route: "play X on YouTube" (NEW #7, 2026-10-02) ─────────
# Live 21:43:17 (2026-10-01): Whisper wrote "Jarvis plays <artist> essentials
# on YouTube". ONE command never reaches the chain rules below (they need two
# or more segments), so the model chose: it emitted [ACTION: youtube] - the
# SEARCH action - the results page opened, nothing played, and JARVIS said
# "Right away, sir." The monolith's built-in utterance route
# (_utterance_route_reply) asks THIS function first, so a whole "play / plays
# / put on <X> on YouTube" request runs youtube_play without the model. A
# search, a vague object ("play it on YouTube"), another service, or an extra
# clause ("... on the left monitor") is NOT claimed - the model keeps those.
_YT_WAKE_LEAD_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)[\s,]+)?jarvis\b[\s,.:;!?-]*", re.IGNORECASE)
_YT_PLAY_RE = re.compile(
    r"^(?:play|plays|put\s+on)\s+(.+?)\s+(?:on|from)\s+(?:the\s+)?"
    r"(?:you\s?tube|yt)(?:[\s,]+(?:please|for\s+me|now))?$", re.IGNORECASE)
# Objects that only make sense against earlier context: the model, which
# sees the conversation, resolves those - a literal YouTube search for "it"
# would play a random video.
_YT_VAGUE_OBJECTS = frozenset({
    "it", "that", "this", "them", "those", "these", "one", "something",
    "anything", "whatever", "something good", "a video", "the video",
    "a song", "the song", "music", "some music", "that video", "that song",
    "this video", "this song",
})
_YT_ROUTE_MAX_ARG = 150


def youtube_play_route(utterance) -> str | None:
    """``"[ACTION: youtube_play, <X>]"`` for a whole "play <X> on YouTube"
    request (Whisper's "plays" and a leading wake word / "can you" / "please"
    included), else None. Never raises."""
    try:
        if not isinstance(utterance, str) or not utterance.strip():
            return None
        s = _YT_WAKE_LEAD_RE.sub("", utterance, count=1)
        s = _strip_lead_filler(s)
        s = " ".join(_strip(s).split())
        m = _YT_PLAY_RE.match(s)
        if not m:
            return None
        query = " ".join(_strip_play_filler(_strip(m.group(1) or "")).split())
        # "play THAT MrBeast video on YouTube" points at something already
        # on screen (2026-10-05, live 00:28:23): never a new search - the
        # screen route / the brain's click takes it.
        if re.match(r"^(?:that|this|the\s+one)\b", query, re.IGNORECASE):
            return None
        if (not query or query.lower() in _YT_VAGUE_OBJECTS
                or len(query) > _YT_ROUTE_MAX_ARG
                or any(c in query for c in "[]\r\n")):
            return None
        return f"[ACTION: youtube_play, {query}]"
    except Exception:
        return None


# ── Screen routes (2026-10-05) ────────────────────────────────────────────
# The monolith's _utterance_route_reply asks THIS first (before the YouTube
# route): pure text over core.onscreen_refs, with the moment's state passed
# in by the caller (an open "which one?", a recent JARVIS UI action, whether
# the screen watcher runs). Each claim is a token the parse loop runs like
# any other, so the guards still apply.
#   * the answer to JARVIS's own "which one?" (<= 90 s): "the second one",
#     "the burger one", a plain "yes" only when one option was offered and no
#     other confirmation / offer is open        -> click_on_screen, pick:<n>
#   * within 120 s of a JARVIS UI action: "not that one" / "that's not the
#     right video"                              -> undo_click, other
#     "go back" / "undo that"                   -> undo_click
#   * "the one that was on screen at the time" (a scene to read it from)
#                                               -> click_on_screen, scene:previous
#   * "tell Claude to ..." (a developer note)   -> note_for_claude, <note>
#   * "stop watching (for N minutes)" / "don't watch this" / "you can watch
#     again" / "are you watching?"              -> screen_memory, ...
#   * "forget the last hour of what you saw"    -> forget_screen, <span>
#   * a WHOLE "click that X" / "play that X video" (no second command)
#                                               -> click_on_screen, <X>
def _token_arg(text) -> str:
    """An action argument with no brackets / newlines (they would end or
    break the token)."""
    return " ".join(str(text or "").replace("[", "(").replace("]", ")")
                    .split())[:300]


def screen_route(utterance, state=None) -> str | None:
    """The token a screen route claims ``utterance`` with, else None (see
    the block comment). ``state`` keys: pending (core.grounded_click's open
    question or None), allow_yes, recent_ui (core.grounded_click.undoable),
    watching, other_watch, scenes, claude_note, click, click_route, and
    app_known (name -> True when an open window is that app). Never
    raises."""
    try:
        from core import onscreen_refs as _or
        st = dict(state or {})
        if not isinstance(utterance, str) or not _or.clean(utterance):
            return None
        click_ok = st.get("click", True)
        p = st.get("pending")
        if p and click_ok:
            idx = _or.pending_choice_answer(utterance, p.get("options") or (),
                                            allow_yes=bool(st.get("allow_yes")))
            if idx is not None:
                return f"[ACTION: click_on_screen, pick:{idx + 1}]"
        if st.get("recent_ui") or p:
            c = _or.is_ui_correction(utterance)
            if c == "other":
                return "[ACTION: undo_click, other]"
            if c == "undo" and st.get("recent_ui"):
                return "[ACTION: undo_click]"
        if click_ok and _or.is_scene_back_reference(utterance) and (
                st.get("recent_ui") or p or st.get("scenes")):
            return "[ACTION: click_on_screen, scene:previous]"
        if st.get("claude_note", True):
            note = _or.claude_note(utterance)
            if note:
                return f"[ACTION: note_for_claude, {_token_arg(note)}]"
        cmd = _or.screen_memory_command(utterance)
        if cmd:
            op = cmd["op"]
            bare = not re.search(r"\b(?:screens?|monitors?)\b",
                                 _or.clean(utterance), re.IGNORECASE)
            if op == "pause":
                if bare and (not st.get("watching") or st.get("other_watch")):
                    return None
                mins = cmd.get("minutes")
                return ("[ACTION: screen_memory, pause"
                        + (f" {mins:g}" if mins else "") + "]")
            if op == "resume":
                return "[ACTION: screen_memory, unpause]"
            if op == "status":
                return "[ACTION: screen_memory, status]"
            if op == "exclude_this":
                return "[ACTION: screen_memory, exclude_this]"
            if op == "exclude_app":
                # "stop watching the room" (guard mode), "stop watching the
                # video" are not about screen memory (review 2026-10-05):
                # an app exclusion is claimed only while screen memory runs
                # AND the words name an app that is open on the screen
                # (``app_known``); anything else is the brain's.
                known = st.get("app_known")
                if not (st.get("watching") and callable(known)
                        and known(cmd["app"])):
                    return None
                return (f"[ACTION: screen_memory, exclude "
                        f"{_token_arg(cmd['app'])}]")
        span = _or.forget_span(utterance)
        if span:
            if span.get("all"):
                arg = "all"
            elif span.get("today"):
                arg = "today"
            else:
                arg = f"{float(span['seconds']) / 60.0:g}m"
            return f"[ACTION: forget_screen, {arg}]"
        if click_ok and st.get("click_route", True):
            target = _or.onscreen_click_target(utterance)
            if target:
                return f"[ACTION: click_on_screen, {_token_arg(target)}]"
    except Exception:
        return None
    return None


# ── Single-command route: "close all windows except X" (2026-10-03) ─────────
# Live 17:22: a request to close every window but the Claude app went to the
# model, which had no action for it: it ran list_windows and then
# minimize_window six times - JARVIS's own HUD and Reticle and two shell
# windows among them - and closed nothing. The monolith's built-in utterance
# route (_utterance_route_reply) asks THIS function, so a whole "close /
# minimize all windows except X" request runs close_all_windows_except /
# minimize_all_windows_except with the owner's own names, never a minimize the
# model improvised. The token still goes through parse_and_run_actions, so the
# "Close N windows, sir? Say yes." pushback applies. Deliberately NOT one of
# the chain rules below: the chain resolver and Controlled mode run their
# steps directly, which would skip that confirmation.
_WK_VERB_RE = (r"(?P<verb>close(?:\s+(?:out|down))?|closed|shut(?:\s+down)?|"
               r"minimi[sz]e|hide)")
_WK_OBJECT_RE = (
    r"(?:all(?:\s+of)?(?:\s+(?:the|my))?(?:\s+(?:other|open))*"
    r"(?:\s+(?:windows?|apps?|applications?|programs?))?"
    r"|every(?:thing(?:\s+else)?|(?:\s+(?:other|open))*\s+"
    r"(?:window|app|application|program)))"
    r"(?:\s+(?:that\s+(?:are|is)\s+)?open)?")
_WK_EXCEPT_RE = (r"(?:except(?:\s+for)?|but(?:\s+not)?|other\s+than|"
                 r"apart\s+from|besides|save\s+for|excluding)")
_WK_KEEP_VERB_RE = r"(?:(?:and|but)\s+)?(?:keep|keeping|leave|leaving)"
_WINDOW_KEEP_RE = re.compile(
    r"^" + _WK_VERB_RE + r"\s+" + _WK_OBJECT_RE + r"[\s,]+"
    r"(?:" + _WK_EXCEPT_RE + r"|" + _WK_KEEP_VERB_RE + r")\s+"
    r"(?P<keep>.+?)"
    r"(?:\s+(?:open|alone|running))?"
    r"(?:[\s,]+(?:please|for\s+me|now))*$", re.IGNORECASE)
# A keep that only makes sense against context, or names nothing: the model
# resolves those. ("this one" / "the current window" are fine: the action
# keeps the window in front.)
_WK_VAGUE_RE = re.compile(
    r"\b(?:it|them|those|these|you|your|yours|yourself|which|whatever|"
    r"something|anything)\b|^(?:the\s+)?one$", re.IGNORECASE)
_WK_MAX_KEEP = 120
# A trailing vocative or thanks is not a name to keep (review 2026-10-03:
# "..., Jarvis" kept every window of the owner's titled with the word, and
# "thanks" was reported as "I saw no thanks window").
_WK_TRAIL_RE = re.compile(
    r"(?:[\s,]+(?:jarvis|sir|thanks|thank\s+you|cheers))+[\s.!?]*$",
    re.IGNORECASE)
# A second command riding in the keep ("... except Claude, minimize Spotify",
# "... and focus Chrome"): the chain splitter knows no window verbs, so the
# command was swallowed as a name to keep (review 2026-10-03). The model
# takes such a turn.
_WK_SECOND_COMMAND_RE = re.compile(
    r"(?:^|[,;&]|\band\b|\bplus\b)\s*(?:then\s+)?"
    r"(?:minimi[sz]e|maximi[sz]e|hide|show|focus|move|snap|restore|open|"
    r"launch|start|play|close|closed|shut|turn|set|put|mute|unmute|pause|"
    r"resume|stop|kill|quit|exit|switch|bring|search|find|take|send|tell|"
    r"read|remind|make|dim|skip)\b", re.IGNORECASE)


def window_keep_route(utterance) -> str | None:
    """``"[ACTION: close_all_windows_except, <keep>]"`` (or
    ``minimize_all_windows_except`` for "minimize" / "hide") for a whole
    "close all windows except X" request - "everything but X", "all apps
    except for X and Y", "all windows and keep X open" - with a leading wake
    word / "can you" / "please"; else None. A keep that also carries a second
    command ("... except Claude and play jazz") is left to the model. Never
    raises."""
    try:
        if not isinstance(utterance, str) or not utterance.strip():
            return None
        s = _YT_WAKE_LEAD_RE.sub("", utterance, count=1)
        s = _strip_lead_filler(s)
        s = " ".join(_strip(s).split())
        s = _strip(_WK_TRAIL_RE.sub("", s))
        m = _WINDOW_KEEP_RE.match(s)
        if not m:
            return None
        keep = " ".join(_strip(m.group("keep") or "").split())
        # "... but leave Claude open": the "but" took the except slot.
        keep = re.sub(r"^(?:keep|keeping|leave|leaving)\s+", "", keep,
                      flags=re.IGNORECASE)
        # "... but don't close Claude": the name is what follows.
        keep = re.sub(r"^(?:do\s*n[o']?t|do\s+not)\s+(?:close|touch|"
                      r"minimi[sz]e|hide)\s+", "", keep, flags=re.IGNORECASE)
        if (not keep or len(keep) > _WK_MAX_KEEP
                or any(c in keep for c in "[]\r\n")
                or _WK_VAGUE_RE.search(keep)
                or _WK_SECOND_COMMAND_RE.search(keep)
                or len(_split_chain(keep)) > 1
                or re.search(r"\bthen\b", keep, re.IGNORECASE)):
            return None
        verb = (m.group("verb") or "").lower()
        action = ("minimize_all_windows_except"
                  if verb.startswith(("minimi", "hide"))
                  else "close_all_windows_except")
        return f"[ACTION: {action}, {keep}]"
    except Exception:
        return None


# Map common spoken units to seconds (used by both timer and focus rules).
_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}


def _arg_focus_duration(m: re.Match) -> str:
    """Build a duration string the focus_mode action understands.

    focus_mode accepts strings like '45 minutes' or '2 hours' (it has its
    own parser). If the user didn't give a number, return '' so the
    action falls back to its DEFAULT_DURATION_SECONDS.
    """
    n = m.group(1) if m.lastindex and m.lastindex >= 1 else None
    unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
    if not n or not unit:
        return ""
    unit = unit.lower()
    # Normalize 'min' / 'hr' to their long forms.
    long_unit = {"min": "minutes", "mins": "minutes",
                 "hr": "hours", "hrs": "hours"}.get(unit, unit)
    if not long_unit.endswith("s"):
        long_unit = long_unit + "s"
    return f"{n} {long_unit}"


def _arg_set_timer(m: re.Match) -> str:
    """Build the 'duration | message' string set_timer wants.

    Patterns capture (number, unit, optional_message).
    """
    n = m.group(1) if m.lastindex and m.lastindex >= 1 else None
    unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
    msg = m.group(3) if m.lastindex and m.lastindex >= 3 else None
    if not n or not unit:
        return ""
    long_unit = {"min": "minutes", "mins": "minutes",
                 "hr": "hours", "hrs": "hours"}.get(unit.lower(), unit.lower())
    if not long_unit.endswith("s"):
        long_unit = long_unit + "s"
    message = (msg or "").strip() or "timer"
    return f"{n} {long_unit} | {message}"


# Each rule: (regex_list, action_name, arg_fn, confirmation_phrase, [aliases])
_INTENT_RULES: list[dict] = [
    # ── Streaming: 'play X on Netflix' (must come BEFORE the music rule,
    #    which would otherwise take 'X on Netflix' as a song) ──────────
    {
        "patterns": [
            r"^(?:play|put\s+on|queue(?:\s+up)?)\s+(.+?)\s+on\s+"
            r"(" + _STREAMING_SERVICE_ALT + r")$",
        ],
        "action": "play_streaming",
        "arg_fn": _arg_play_streaming,
        "confirmation": "playback started",
    },
    # ── Music playback ───────────────────────────────────────────────
    {
        "patterns": [
            r"^play\s+(.+)$",
            r"^put\s+on\s+(.+)$",
            r"^queue\s+(?:up\s+)?(.+)$",
        ],
        "action": "play_music",
        # Fallbacks if iTunes/play_music isn't available but a streaming
        # service action is.
        "fallbacks": ["apple_music", "spotify", "play_streaming"],
        "arg_fn": _arg_play_music,
        "confirmation": "music queued",
    },
    {
        "patterns": [
            r"^pause(?:\s+(?:the\s+)?(?:music|song|track|it|that))?$",
            r"^stop(?:\s+(?:the\s+)?(?:music|song|track))$",
        ],
        "action": "pause_music",
        "fallbacks": ["media_playpause"],
        "arg_fn": lambda m: "",
        "confirmation": "music paused",
    },
    {
        "patterns": [
            r"^resume(?:\s+(?:the\s+)?(?:music|song|track))?$",
            r"^(?:un[- ]?pause|continue)(?:\s+(?:the\s+)?(?:music|song|track))?$",
        ],
        "action": "resume_music",
        "fallbacks": ["media_playpause"],
        "arg_fn": lambda m: "",
        "confirmation": "music resumed",
    },
    {
        "patterns": [
            r"^(?:next|skip)(?:\s+(?:this\s+)?(?:song|track|one))?$",
            r"^skip\s+(?:this|it|ahead)$",
        ],
        "action": "next_song",
        "fallbacks": ["media_next"],
        "arg_fn": lambda m: "",
        "confirmation": "skipped to next track",
    },
    {
        "patterns": [
            r"^(?:previous|prev|last)(?:\s+(?:song|track))?$",
            r"^go\s+back(?:\s+(?:a\s+)?(?:song|track))?$",
        ],
        "action": "previous_song",
        "fallbacks": ["media_prev"],
        "arg_fn": lambda m: "",
        "confirmation": "back one track",
    },

    # ── Focus mode (must come BEFORE the generic timer rule so
    #     'start a 45 minute focus timer' routes to focus_mode, not
    #     set_timer) ────────────────────────────────────────────────
    {
        "patterns": [
            r"^(?:start|begin|engage|turn\s+on|activate|enter|kick\s+off)\s+"
            r"(?:a\s+|an\s+)?"
            r"(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)\s+"
            r"focus(?:\s+(?:mode|timer|session|block))?$",
            r"^(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)\s+"
            r"focus(?:\s+(?:mode|timer|session|block))?$",
        ],
        "action": "focus_mode",
        "arg_fn": _arg_focus_duration,
        "confirmation": "focus mode armed",
    },
    {
        "patterns": [
            r"^(?:start|begin|engage|turn\s+on|activate|enter|kick\s+off)\s+"
            r"(?:a\s+|an\s+)?focus(?:\s+(?:mode|session|block))?$",
            r"^focus\s+mode$",
            r"^do\s+not\s+disturb$",
            r"^(?:start|begin|engage)\s+(?:a\s+)?(?:dnd|d\.n\.d\.)$",
        ],
        "action": "focus_mode",
        "arg_fn": lambda m: "",
        "confirmation": "focus mode armed",
    },

    # ── Generic timer ─────────────────────────────────────────────────
    {
        "patterns": [
            r"^(?:set|start|begin|kick\s+off)\s+(?:a\s+|an\s+)?"
            r"(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)"
            r"\s+timer(?:\s+(?:for|to)\s+(.+))?$",
            r"^(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)\s+timer$",
            r"^remind\s+me\s+in\s+(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)"
            r"(?:\s+to\s+(.+))?$",
        ],
        "action": "set_timer",
        "arg_fn": _arg_set_timer,
        "confirmation": "timer set",
    },

    # ── Volume ────────────────────────────────────────────────────────
    {
        "patterns": [
            r"^(?:turn\s+(?:it\s+|the\s+(?:volume|sound)\s+)?up|"
            r"volume\s+up|louder|crank\s+(?:it|the\s+volume)\s+up)$",
        ],
        "action": "volume_up",
        "arg_fn": lambda m: "",
        "confirmation": "volume up",
    },
    {
        "patterns": [
            r"^(?:turn\s+(?:it\s+|the\s+(?:volume|sound)\s+)?down|"
            r"volume\s+down|quieter|lower\s+(?:it|the\s+volume))$",
        ],
        "action": "volume_down",
        "arg_fn": lambda m: "",
        "confirmation": "volume down",
    },
    {
        "patterns": [
            r"^(?:mute(?:\s+(?:it|that|the\s+(?:sound|music|volume)))?|silence(?:\s+(?:it|that))?)$",
        ],
        "action": "volume_mute",
        "arg_fn": lambda m: "",
        "confirmation": "muted",
    },
    {
        # 2026-10-01: volume_mute now SETS mute instead of toggling it, so
        # "unmute" needs its own deterministic route (it used to reach the
        # toggle). Anchored like the mute rule: "unmute" never matches it.
        "patterns": [
            r"^(?:unmute(?:\s+(?:it|that|the\s+(?:sound|music|volume|audio)))?|"
            r"turn\s+the\s+(?:sound|audio|volume)\s+back\s+on)$",
        ],
        "action": "volume_unmute",
        "arg_fn": lambda m: "",
        "confirmation": "unmuted",
    },

    # ── Screenshot ────────────────────────────────────────────────────
    {
        "patterns": [
            r"^(?:take\s+(?:a\s+)?screenshot|screenshot(?:\s+(?:the\s+)?screen)?|"
            r"capture\s+(?:the\s+)?screen|grab\s+(?:a\s+|the\s+)?screen(?:shot)?)$",
        ],
        "action": "screenshot",
        "arg_fn": lambda m: "",
        "confirmation": "screenshot captured",
    },

    # ── Task queue ────────────────────────────────────────────────────
    {
        "patterns": [
            r"^(?:show|list|read)\s+(?:my\s+)?tasks?$",
            r"^what(?:'s|\s+is)\s+(?:on\s+)?(?:my|the)\s+(?:task\s+list|to-?do(?:\s+list)?)$",
        ],
        "action": "show_tasks",
        "arg_fn": lambda m: "",
        "confirmation": "tasks shown",
    },
]


# Compile patterns once at module load.
for _rule in _INTENT_RULES:
    _rule["_compiled"] = [re.compile(p, re.IGNORECASE) for p in _rule["patterns"]]


# ──────────────────────────────────────────────────────────────────────────
#  SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────
# Split the utterance on chain conjunctions. The split is conservative:
# only conjunctions surrounded by word boundaries split, and the result is
# only treated as a chain if ≥2 segments survive normalization.

# Two-tier separators. Tier-strong markers are very unlikely to appear
# inside an entity name ("Earth Wind and Fire"), so we split on them
# eagerly. Tier-weak (bare " and ") is ambiguous, so we only split there
# when the right-hand side looks like a fresh command (starts with a
# command verb).
_STRONG_SEP_RE = re.compile(
    r"(?:"
    r",\s+(?:and\s+then|and|then|also|plus)\s+"
    r"|;\s+(?:and\s+)?(?:then\s+)?"
    r"|\s+and\s+then\s+"
    r"|\s+then\s+"
    r"|\s+also\s+"
    r"|\s+plus\s+"
    # NOTE: the bare `,\s+` alternative was REMOVED (2026-07-14 bug-hunt #10).
    # It split eagerly on ANY comma, so "play Earth, Wind and Fire" and "set a
    # timer for 1, 2, 3" and "remind me to call Bob, Jr." tore a comma-bearing
    # ENTITY into a bogus second action. A comma followed by a real chain
    # connector (line 1 above: ", and then …") is still a strong marker; a BARE
    # comma is now gated by _split_on_comma, exactly like bare " and ".
    r")",
    re.IGNORECASE,
)

# Bare comma — split only where the right-hand side opens with a command verb.
_COMMA_SEP_RE = re.compile(r",\s+")

# Bare " and " — only used after a stronger marker has confirmed the
# utterance is structurally a chain, OR the right-hand side begins with
# a recognized command verb.
_AND_SEP_RE = re.compile(r"\s+and\s+", re.IGNORECASE)

# Words that, when they open a segment, signal "this is a fresh command,
# not a continuation of the previous entity." Used to gate bare " and "
# splitting against false positives like "Michael Jackson and the
# Jackson 5".
_COMMAND_VERBS = {
    "activate", "begin", "capture", "close", "crank", "dim", "do",
    "engage", "find", "go", "grab", "kick", "launch", "list", "lower",
    "louder", "make", "mute", "next", "open", "pause", "play", "previous",
    "put", "queue", "quieter", "read", "remind", "resume", "screenshot",
    "search", "set", "show", "silence", "skip", "start", "stop", "take",
    "tell", "turn", "unpause", "volume",
}


def _looks_like_command_start(segment: str) -> bool:
    """Does this segment open with a known command verb?"""
    seg = segment.strip().lower()
    if not seg:
        return False
    first = seg.split(None, 1)[0]
    # Strip leading articles users sometimes drop in front of a verb
    # ("the volume up"). If the first token is an article, peek at the
    # second.
    if first in {"the", "a", "an"} and " " in seg:
        first = seg.split(None, 2)[1]
    return first in _COMMAND_VERBS

# Filler / lead-ins are stripped from the beginning of the WHOLE utterance by
# the shared core.lead_fillers.strip_lead_filler (imported above as
# _strip_lead_filler). This module used to carry its own stale copy — no comma
# wake variants ("jarvis, "), single pass — so Controlled mode refused
# "JARVIS, take a screenshot" (2026-07-21 audit). ONE implementation now.


def _split_on_and(chunk: str) -> list[str]:
    """Split `chunk` on bare ' and ', but only at boundaries where the
    right-hand side opens with a recognized command verb. This stops
    'Michael Jackson and the Jackson 5' / 'Earth Wind and Fire' from
    being torn apart while still catching 'play X and start Y'."""
    pieces: list[str] = []
    last = 0
    for m in _AND_SEP_RE.finditer(chunk):
        rhs = chunk[m.end():]
        if _looks_like_command_start(rhs):
            pieces.append(chunk[last:m.start()])
            last = m.end()
    pieces.append(chunk[last:])
    return pieces


def _split_on_comma(chunk: str) -> list[str]:
    """Split `chunk` on a bare ', ', but only where the right-hand side opens
    with a recognized command verb — the same gate _split_on_and uses. This
    keeps 'play Earth, Wind and Fire' and 'call Bob, Jr.' intact while still
    catching 'dim the lights, play jazz'. 2026-07-14 bug-hunt #10."""
    pieces: list[str] = []
    last = 0
    for m in _COMMA_SEP_RE.finditer(chunk):
        rhs = chunk[m.end():]
        if _looks_like_command_start(rhs):
            pieces.append(chunk[last:m.start()])
            last = m.end()
    pieces.append(chunk[last:])
    return pieces


def _split_chain(utterance: str) -> list[str]:
    """Split into segments using strong separators eagerly, then bare
    ' and ' only at command-verb boundaries.

    Returns trimmed segments. A single-segment result means no chain
    was detected.
    """
    s = _strip_lead_filler(utterance)
    # Trim trailing punctuation that the LLM / Whisper sometimes adds.
    s = _PUNCT_TAIL.sub("", s)

    strong = _STRONG_SEP_RE.split(s)
    expanded: list[str] = []
    for chunk in strong:
        for comma_piece in _split_on_comma(chunk):
            for piece in _split_on_and(comma_piece):
                expanded.append(piece)
    return [p.strip() for p in expanded if p and p.strip()]


# ──────────────────────────────────────────────────────────────────────────
#  MATCHING
# ──────────────────────────────────────────────────────────────────────────

def _resolve_action(rule: dict, available_actions: Iterable[str]) -> str | None:
    """Return the first action name (primary or fallback) that's registered."""
    actions = set(available_actions)
    primary = rule.get("action")
    if primary and primary in actions:
        return primary
    for fb in rule.get("fallbacks", []) or []:
        if fb in actions:
            return fb
    return None


def _match_segment(segment: str, available_actions: Iterable[str]) -> ChainStep | None:
    """Try every rule against `segment`. Return the first match whose action
    is actually registered, else None."""
    seg = _strip(segment)
    for rule in _INTENT_RULES:
        action_name = _resolve_action(rule, available_actions)
        if not action_name:
            continue
        for pat in rule["_compiled"]:
            m = pat.match(seg)
            if not m:
                continue
            try:
                arg = rule["arg_fn"](m)
            except Exception:
                arg = ""
            return ChainStep(
                action=action_name,
                arg=arg,
                confirmation=rule["confirmation"],
                source=segment,
            )
    return None


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────

def match_single_intent(
    utterance: str,
    available_actions: Iterable[str],
) -> ChainStep | None:
    """Try to match `utterance` against the intent rules as a single command.

    Same rule set as the chain resolver, but treats the entire utterance
    as one segment instead of splitting on chain separators. Filler
    lead-ins ("could you", "please", "jarvis,") and trailing punctuation
    are stripped before matching.

    Returns the matched ChainStep (action name, arg string, confirmation
    phrase) when the utterance maps cleanly onto one rule, else None.
    Used by core.mode_router for Controlled mode dispatch, where the
    user wants deterministic skill matching with no LLM in the loop.
    """
    if not utterance or not utterance.strip():
        return None
    s = _strip_lead_filler(utterance)
    s = _PUNCT_TAIL.sub("", s).strip()
    if not s:
        return None
    return _match_segment(s, available_actions)


def command_chain_resolver(
    utterance: str,
    available_actions: Iterable[str],
) -> ChainResult | None:
    """Detect multi-step intents in `utterance`.

    Returns a ChainResult only when:
      • the utterance splits into ≥2 segments via a chain separator, AND
      • at least 2 segments match known intent rules whose target action
        is in `available_actions`.

    Otherwise returns None (caller should fall through to the LLM).
    """
    if not utterance or not utterance.strip():
        return None

    segments = _split_chain(utterance)
    if len(segments) < 2:
        return None

    steps: list[ChainStep] = []
    unknown: list[str] = []
    for seg in segments:
        step = _match_segment(seg, available_actions)
        if step is not None:
            steps.append(step)
        else:
            unknown.append(seg)

    # Conservative: require at least 2 matched steps to treat this as a
    # chain. One match + one unknown could just be a normal sentence the
    # LLM should handle.
    if len(steps) < 2:
        return None

    return ChainResult(steps=steps, unknown=unknown)


# Words used to count chained steps in the consolidated confirmation.
_COUNT_WORDS = {
    2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
}


# Canonical marker list lives in core/failure_markers.py and is shared with
# bobert_companion._is_failure so the two can't drift. Actions return free-text
# strings; these substrings (case-insensitive) flag a result as a failure even
# though the call didn't raise.
_FAIL_MARKERS = FAILURE_MARKERS


def _is_failure_result(result) -> bool:
    if not isinstance(result, str) or not result:
        return False
    lower = result.lower()
    return any(m in lower for m in _FAIL_MARKERS)


# Honest NO-OP results: not errors, but the step did nothing, so the rule's
# success phrase would be false. "pause the music and turn it down" with
# nothing playing said "music paused" (2026-10-01). Chain-only on purpose:
# FAILURE_MARKERS also drives the monolith's failure re-prompt, and the
# single-command pause_music path already voices these results as they are.
_CHAIN_NOOP_MARKERS: dict[str, str] = {
    "nothing seems to be playing": "nothing was playing",
    "pyautogui unavailable": "media keys unavailable",
}


def _noop_phrase(result) -> str | None:
    """The consolidated-line phrase for a no-op step result, else None."""
    if not isinstance(result, str) or not result:
        return None
    low = result.lower()
    for marker, phrase in _CHAIN_NOOP_MARKERS.items():
        if marker in low:
            return phrase
    return None


def _failure_phrase(result: str, fallback: str) -> str:
    """Compress a failure result into one short phrase for the consolidated reply."""
    if not isinstance(result, str) or not result.strip():
        return f"{fallback} failed"
    s = result.strip().split("\n", 1)[0]
    s = re.split(r"[.!?](?:\s|$)", s, maxsplit=1)[0].strip()
    if not s:
        return f"{fallback} failed"
    s = _PUNCT_TAIL.sub("", s)
    if s[:1].isupper() and not s.startswith(("JARVIS", "J.A.R.V.I.S.")):
        s = s[0].lower() + s[1:]
    if len(s) > 80:
        s = s[:77].rstrip() + "..."
    return s


def _format_consolidated(steps: list[ChainStep], unknown: list[str]) -> str:
    """Build the single TTS line summarizing what got dispatched.

    Example output:
      "Three things, sir: music queued, focus mode armed, timer set."
    """
    n = len(steps)
    word = _COUNT_WORDS.get(n, str(n))
    confirmations = [s.confirmation for s in steps]
    line = f"{word} things, sir: " + ", ".join(confirmations) + "."
    if unknown:
        # Note dropped segments succinctly — keeps the user informed
        # without dumping them into a chatty multi-sentence reply.
        if len(unknown) == 1:
            line += f" I didn't catch '{unknown[0]}', though."
        else:
            line += f" {len(unknown)} other items I couldn't place."
    return line


def resolve_and_dispatch(
    utterance: str,
    actions: dict[str, Callable[[str], str]],
    is_self_voiced: Callable[[str], bool] | None = None,
) -> str | None:
    """One-call entry point used by the main loop.

    Resolves the chain and, if successful, runs each step against the
    `actions` dict in order. Returns the consolidated TTS confirmation
    string, or None if no chain was detected.

    Action exceptions are caught per-step so one failing step never
    aborts the rest of the chain. A failed step is reported in the
    consolidated line as 'X failed'.

    ``is_self_voiced(name)``: an action that does all of its own talking
    (a device dialogue) is never run as a chain step: it is skipped with a
    log line, like an unregistered action. Default: nothing is.
    """
    result = command_chain_resolver(utterance, actions.keys())
    if result is None:
        return None

    # Re-check action availability BEFORE executing anything. An action can be
    # de-registered between resolve and dispatch (actions.get() -> None). If we
    # discover that mid-execution and then bail with <2 survivors, the caller
    # treats None as 'no chain' and re-runs the surviving command through the
    # LLM path — double-executing it (timer started twice, volume applied
    # twice). Resolving availability up front means that when we bail, NOTHING
    # has run yet, so the LLM fall-through is safe.
    runnable: list[tuple[ChainStep, Callable[[str], str]]] = []
    for step in result.steps:
        if is_self_voiced is not None:
            try:
                _sv = bool(is_self_voiced(step.action))
            except Exception:
                _sv = False
            if _sv:
                print(f"  [chain] skipped self-voiced action {step.action}")
                result.unknown.append(step.source)
                continue
        fn = actions.get(step.action)
        if fn is None:
            # Race: action was registered when we resolved but isn't now.
            # Demote to unknown.
            result.unknown.append(step.source)
            continue
        runnable.append((step, fn))

    if len(runnable) < 2:
        # Too few survivors to be a chain. Bail BEFORE executing so the caller
        # can safely fall through to the LLM with nothing double-dispatched.
        return None

    dispatched: list[ChainStep] = []
    for step, fn in runnable:
        try:
            rv = fn(step.arg)
            _noop = _noop_phrase(rv)
            if _is_failure_result(rv) or _noop:
                # Action ran without raising but returned a failure marker
                # (or an honest "did nothing") — surface that instead of the
                # success confirmation so the user hears what really happened.
                dispatched.append(ChainStep(
                    action=step.action,
                    arg=step.arg,
                    confirmation=_noop or _failure_phrase(rv, step.confirmation),
                    source=step.source,
                ))
            else:
                dispatched.append(step)
        except Exception as e:
            # Keep the step's slot but flag the failure in the
            # confirmation so the user knows it didn't land.
            dispatched.append(ChainStep(
                action=step.action,
                arg=step.arg,
                confirmation=f"{step.confirmation} failed ({type(e).__name__})",
                source=step.source,
            ))

    # Every runnable step appends exactly one entry above (success, failure
    # marker, or caught exception all append), so len(dispatched) == len(runnable)
    # >= 2 here. We must NOT return None once actions have executed: that would
    # let the caller re-run them via the LLM. The <2 floor now lives in the
    # pre-execution availability check above.
    return _format_consolidated(dispatched, result.unknown)
