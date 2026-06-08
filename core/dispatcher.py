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

from core.failure_markers import FAILURE_MARKERS


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


def _arg_play_music(m: re.Match) -> str:
    """Extract artist/song from 'play X' / 'put on X' / 'queue X'.

    If the user said 'play some <X>' or 'play me <X>', drop the filler so
    iTunes search gets the bare query.
    """
    q = _arg_first_group(m).lower()
    for filler in ("some ", "me some ", "me "):
        if q.startswith(filler):
            q = q[len(filler):]
            break
    # Preserve original casing where possible: re-extract and apply the
    # same prefix trim to the cased string.
    raw = _arg_first_group(m)
    raw_low = raw.lower()
    for filler in ("some ", "me some ", "me "):
        if raw_low.startswith(filler):
            raw = raw[len(filler):]
            break
    return raw.strip()


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
    r"|,\s+"
    r")",
    re.IGNORECASE,
)

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

# Filler / lead-ins to strip from the beginning of the WHOLE utterance.
_LEAD_FILLERS = [
    "could you ", "can you ", "would you ", "please ",
    "jarvis ", "hey jarvis ", "ok jarvis ", "okay jarvis ",
    "go ahead and ", "i need you to ", "i'd like you to ",
]


def _strip_lead_filler(s: str) -> str:
    low = s.lower().lstrip()
    for f in _LEAD_FILLERS:
        if low.startswith(f):
            # Strip the filler from the original (preserving case).
            return s.lstrip()[len(f):].lstrip()
    return s.strip()


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
        for piece in _split_on_and(chunk):
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
) -> str | None:
    """One-call entry point used by the main loop.

    Resolves the chain and, if successful, runs each step against the
    `actions` dict in order. Returns the consolidated TTS confirmation
    string, or None if no chain was detected.

    Action exceptions are caught per-step so one failing step never
    aborts the rest of the chain. A failed step is reported in the
    consolidated line as 'X failed'.
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
            if _is_failure_result(rv):
                # Action ran without raising but returned a failure marker
                # — surface the action's own message instead of the
                # success confirmation so the user hears what went wrong.
                dispatched.append(ChainStep(
                    action=step.action,
                    arg=step.arg,
                    confirmation=_failure_phrase(rv, step.confirmation),
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
