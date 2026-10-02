"""core/instant_actions.py — basic commands the brain is not needed for.

WHY THIS MODULE EXISTS
======================
Instant (no-LLM) answers already cover the date and time, the owner's name,
recall, the self-check and the timer list (core/fast_paths.py and the
monolith's _run_action_shortcut). Volume, media transport, the lights and
"pause the print" still went through the brain: a whole local-LLM round to
emit one obvious action token.

INSTANT_ACTIONS_MODE (core/config.py) closes that gap in two steps:
  "shadow" (default)  the brain still answers. The monolith logs the action
                      this module WOULD have run and, once the brain's own
                      actions have run, whether the brain ran the same one:
                      the precision measurement, scored by
                      tools/instant_actions_report.py.
  "on"                the action runs at once with a short spoken line and
                      the LLM is never called.
  "off"               nothing is matched or logged.
The monolith side is bobert_companion._instant_action_for /
_instant_action_after_run, wired into _run_llm_dispatch_body.

PRECISION BEATS RECALL
======================
A wrong instant action is worse than a slow right one, so ``match`` returns
None unless every check passes:
  * short (<= MAX_WORDS words), no '?', not a question
    (claim_validator.looks_like_question) and not a polite ask ("can you
    ...", "would you mind ..."): the brain answers those;
  * ONE command: no chain (dispatcher.command_chain_resolver), no connective
    word (and / then / also / but / or / ...), no comma or semicolon left
    once the wake word and a trailing "please" are stripped;
  * no pronoun ("pause it", "turn it up": what "it" is, is the brain's call)
    and no negation ("don't ...");
  * an anchored rule matches the WHOLE command: the dispatcher's volume and
    media rules (the ones Controlled mode runs with no LLM), through
    turn_checker's clear-command guard so "next" / "continue" / "skip" alone
    never count, or this module's lights on/off and printer-pause rules;
  * the action is in the owner's allowlist AND registered, core.action_risk
    has no confirm rule for it, and the caller's ``blocked(name, arg)`` (the
    monolith's confirmation / pushback / protected-action gates) does not
    object.
The allowlist can only NARROW: a name the rules below never produce
(RULE_ACTIONS) is ignored, so a hand-edited setting cannot widen what runs.

THE LOG
=======
data/instant_actions.jsonl (gitignored), one JSON object per instant turn:
  {"ts": 1759400000.0, "mode": "shadow", "action": "pause_music",
   "brain": ["pause_music"], "agree": true}
  {"ts": 1759400000.0, "mode": "on", "action": "volume_up", "ok": true}
The time and REGISTERED action names only — never the owner's words, an
action's argument or its result. At LOG_MAX_BYTES it is moved to
instant_actions.jsonl.1 (one generation kept) and a new file is started.

Stdlib plus repo modules that do no I/O at import, so it runs on the light
CI tier (tests/test_instant_actions.py).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from core import action_risk as _action_risk
from core import claim_validator as _claim_validator
from core import dispatcher as _dispatcher
from core import turn_checker as _turn_checker
from core.lead_fillers import strip_lead_filler as _strip_lead_filler

__all__ = [
    "DEFAULT_MODE",
    "InstantAction",
    "LOG_NAME",
    "MODES",
    "RULE_ACTIONS",
    "agrees",
    "append_row",
    "brain_ran",
    "match",
    "normalize_allow",
    "normalize_mode",
    "on_row",
    "read_rows",
    "shadow_row",
    "spoken_line",
]

MODES = ("shadow", "on", "off")
DEFAULT_MODE = "shadow"

LOG_NAME = "instant_actions.jsonl"
LOG_MAX_BYTES = 1_000_000

MAX_WORDS = 10
MAX_CHARS = 100

# Every action name a rule below can produce. The owner's allowlist
# (INSTANT_ACTIONS_ALLOW) is intersected with this.
VOLUME_ACTIONS = ("volume_up", "volume_down", "volume_mute", "volume_unmute")
MEDIA_ACTIONS = ("pause_music", "resume_music", "next_song", "previous_song")
LIGHTS_ACTION = "smart_home_control"
PRINTER_PAUSE_ACTION = "pause_print"
_DISPATCHER_ACTIONS = frozenset(VOLUME_ACTIONS + MEDIA_ACTIONS)
RULE_ACTIONS = frozenset(_DISPATCHER_ACTIONS
                         | {LIGHTS_ACTION, PRINTER_PAUSE_ACTION})

_NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")


@dataclass(frozen=True)
class InstantAction:
    action: str     # a registered action name
    arg: str        # its argument: "" except the lights command
    ack: str        # spoken when the action's own result is not a sentence

    @property
    def token(self) -> str:
        """The [ACTION:] token the turn runs, exactly as the brain writes it."""
        if self.arg:
            return f"[ACTION: {self.action}, {self.arg}]"
        return f"[ACTION: {self.action}]"


def normalize_mode(value) -> str:
    """'shadow' / 'on' / 'off'; anything else is the default, 'shadow'
    (which changes nothing the owner hears)."""
    mode = str(value or "").strip().lower()
    return mode if mode in MODES else DEFAULT_MODE


def normalize_allow(value) -> frozenset:
    """The allowlist as lower-case names, narrowed to RULE_ACTIONS. A value
    that is not a list of names allows nothing."""
    if isinstance(value, str) or not isinstance(value, (list, tuple, set,
                                                        frozenset)):
        return frozenset()
    out = set()
    for v in value:
        n = str(v or "").strip().lower()
        if n in RULE_ACTIONS:
            out.add(n)
    return frozenset(out)


# ── the owner's words ─────────────────────────────────────────────────────
_WAKE_LEAD_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)[\s,]+)?jarvis\b[\s,.:;!?-]*", re.IGNORECASE)
_PUNCT_TAIL_RE = re.compile(r"[.!,;:]+\s*$")
# "can you pause the music" / "could you turn off the lights?" / "would you
# mind ..." / "is it possible to ...": a request the brain answers, even
# though claim_validator reads "can you X" as a polite COMMAND.
_POLITE_RE = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|so|and|um+|uh+|well|please|jarvis)[,\s]+)*"
    r"(?:(?:can|could|would|will)(?:n'?t)?\s+(?:you|ya|u)\b"
    r"|(?:do|would)\s+you\s+mind\b"
    r"|is\s+it\s+possible\b|any\s+chance\b|i\s+(?:wonder|was\s+wondering)\b"
    r"|maybe\b|perhaps\b)")
# Words that join or qualify two thoughts: never one plain command.
_CONNECTIVE_RE = re.compile(
    r"\b(?:and|then|also|plus|but|or|after|before|until|unless|while|when|"
    r"if|because|so|instead|except)\b")
# What "it" refers to is the brain's call (the conversation, a pointing arm).
# "this song" / "this track" names its object, so "skip this song" counts.
_PRONOUN_RE = re.compile(
    r"\b(?:it|that|them|those|these|one)\b|\bthis\b(?!\s+(?:song|track)\b)")
_NEGATION_RE = re.compile(r"\b(?:don'?t|dont|do\s+not|never|not|no)\b")

# ── this module's own rules (the dispatcher has none for these) ──────────
_PRINTER_PAUSE_RE = re.compile(
    r"^pause\s+(?:the\s+|my\s+)?(?:3d\s+)?"
    r"(?:print|printer|printing|print\s+job)$")
_LIGHT_ARTICLE = r"(?:the\s+|my\s+|all\s+(?:of\s+)?(?:the\s+|my\s+)?)?"
_LIGHT_QUAL = r"(?P<q>(?:[a-z]+\s+){0,2})"
_LIGHT_NOUN = r"(?:lights?|lamps?)"
_LIGHTS_RES = (
    # turn off the office light / switch on all the lamps
    re.compile(r"^(?:turn|switch)\s+(?P<s>on|off)\s+" + _LIGHT_ARTICLE
               + _LIGHT_QUAL + _LIGHT_NOUN + r"$"),
    # turn the kitchen lights off
    re.compile(r"^(?:turn|switch)\s+" + _LIGHT_ARTICLE + _LIGHT_QUAL
               + _LIGHT_NOUN + r"\s+(?P<s>on|off)$"),
    # lights off / the desk lamp on
    re.compile(r"^" + _LIGHT_ARTICLE + _LIGHT_QUAL + _LIGHT_NOUN
               + r"\s+(?P<s>on|off)$"),
)
# A qualifier is a device or room word ("office", "desk", "night"), never one
# of these.
_LIGHT_QUAL_STOP = frozenset({
    "the", "my", "a", "an", "all", "of", "every", "each", "both", "some",
    "any", "other", "more", "less", "bit", "little", "half", "way", "in",
    "on", "off", "to", "for", "with", "at", "by", "from", "up", "down",
    "back", "again", "now", "please", "sir", "jarvis", "turn", "switch",
    "dim", "dimmer", "bright", "brighter", "light", "lights", "lamp", "lamps",
})


def _clean(text: str) -> str:
    t = str(text or "").replace("’", "'").replace("‘", "'")
    return " ".join(t.split())


def _command_core(text: str) -> str:
    """The command with the wake word, polite lead-ins ("please", "go ahead
    and"), a trailing "please" / "thanks" and end punctuation removed."""
    s = _WAKE_LEAD_RE.sub("", _clean(text), count=1)
    s = _strip_lead_filler(s)
    prev = None
    while prev != s:
        prev = s
        s = _turn_checker._TRAILING_COURTESY_RE.sub("", s)
        s = _PUNCT_TAIL_RE.sub("", s).strip()
    return s


def _lights(core_low: str) -> Optional[str]:
    """'on' / 'off' for a whole lights on/off command, else None."""
    for rx in _LIGHTS_RES:
        m = rx.match(core_low)
        if not m:
            continue
        qual = (m.group("q") or "").split()
        if any(w in _LIGHT_QUAL_STOP for w in qual):
            return None
        return m.group("s")
    return None


def _ack(phrase: str) -> str:
    phrase = phrase.strip()
    return f"{phrase[:1].upper()}{phrase[1:]}, sir." if phrase else "Done, sir."


def match(text, registered: Iterable[str], *, allow,
          blocked: Optional[Callable[[str, str], bool]] = None
          ) -> Optional[InstantAction]:
    """The InstantAction for ``text``, or None (the brain answers).

    ``registered``  the live action registry (the ACTIONS dict works).
    ``allow``       INSTANT_ACTIONS_ALLOW; narrowed to RULE_ACTIONS.
    ``blocked``     optional ``(name, arg) -> bool``: True leaves the turn to
                    the brain (a confirmation, pushback or protected-action
                    gate). A raising check counts as True.
    Never raises."""
    try:
        return _match(text, registered, allow, blocked)
    except Exception:
        return None


def _match(text, registered, allow, blocked):
    if not isinstance(text, str):
        return None
    raw = _clean(text)
    if not raw or len(raw) > MAX_CHARS or "?" in raw:
        return None
    low = raw.lower()
    if _claim_validator.looks_like_question(raw):
        return None
    if _POLITE_RE.match(_WAKE_LEAD_RE.sub("", low, count=1)):
        return None
    names = {str(n).strip().lower() for n in (registered or ())}
    usable = normalize_allow(allow) & names
    if not usable:
        return None
    # Two commands in one breath are the chain resolver's (or the brain's).
    if _dispatcher.command_chain_resolver(raw, names) is not None:
        return None
    core = _command_core(raw)
    core_low = core.lower()
    if (not core or len(core.split()) > MAX_WORDS
            or re.search(r"[,;:\[\]\n]", core_low)
            or _CONNECTIVE_RE.search(core_low)
            or _PRONOUN_RE.search(core_low)
            or _NEGATION_RE.search(core_low)):
        return None

    hit = None
    if _PRINTER_PAUSE_RE.match(core_low):
        if PRINTER_PAUSE_ACTION in usable:
            hit = InstantAction(PRINTER_PAUSE_ACTION, "",
                                "Pausing the print, sir.")
    else:
        state = _lights(core_low)
        if state is not None:
            if LIGHTS_ACTION in usable:
                hit = InstantAction(LIGHTS_ACTION, core,
                                    _ack(f"lights {state}"))
        else:
            step = _dispatcher.match_single_intent(
                core, usable & _DISPATCHER_ACTIONS)
            if (step is not None and not step.arg
                    and step.action in usable
                    and _turn_checker._clear_command(step.source)):
                hit = InstantAction(step.action, "", _ack(step.confirmation))
    if hit is None:
        return None
    if (_action_risk.action_confirm_reason(hit.action)
            or hit.action in _action_risk.SELF_TERMINATING_ACTIONS):
        return None
    if blocked is not None:
        try:
            if blocked(hit.action, hit.arg):
                return None
        except Exception:
            return None
    return hit


# ── what was said and what ran ────────────────────────────────────────────
def spoken_line(hit: InstantAction, result) -> str:
    """What an "on" turn says after its action ran: the action's own result
    when it is already a sentence for the owner ("paused Spotify, sir"), an
    honest no-op phrase when the media keys were missing, else ``hit.ack``
    ("Volume up, sir.")."""
    s = _clean(result) if isinstance(result, str) else ""
    if s and len(s) <= 240 and re.search(r"\bsir\b", s, re.IGNORECASE):
        s = s[:1].upper() + s[1:]
        return s if s[-1] in ".!" else s.rstrip(",;:- ") + "."
    try:
        noop = _dispatcher._noop_phrase(s)
    except Exception:
        noop = None
    if noop:
        return _ack(noop)
    return hit.ack


def brain_ran(action_results, registered: Iterable[str]) -> list:
    """The registered action names that RAN in ``action_results`` (the
    monolith's (name, result, is_informative) list), in order, once each.
    A deferral ("⚠ REQUIRES CONFIRMATION" / pushback / ambiguity), an
    unknown action and a synthetic '_' result did not run."""
    names = {str(n).strip().lower() for n in (registered or ())}
    out: list = []
    for item in action_results or ():
        try:
            n, r = item[0], item[1]
        except Exception:
            continue
        n = str(n or "").strip().lower()
        if (not n or n.startswith("_") or n not in names or n in out
                or not _NAME_RE.match(n)):
            continue
        if isinstance(r, str) and (r.lstrip().startswith("⚠")
                                   or r.startswith("unknown action")):
            continue
        out.append(n)
    return out


def agrees(action: str, brain: Iterable[str],
           same_handler: Optional[Callable[[str, str], bool]] = None) -> bool:
    """True when the brain ran ``action`` itself or an alias of it
    (``same_handler(a, b)``: both names are bound to one handler)."""
    a = str(action or "").strip().lower()
    if not a:
        return False
    for b in brain or ():
        nb = str(b or "").strip().lower()
        if nb == a:
            return True
        if same_handler is not None:
            try:
                if same_handler(a, nb):
                    return True
            except Exception:
                continue
    return False


# ── the log ───────────────────────────────────────────────────────────────
def _names_only(names) -> list:
    return [n for n in (str(x or "").strip().lower() for x in (names or ()))
            if _NAME_RE.match(n)]


def shadow_row(action: str, brain, agree: bool, ts=None) -> dict:
    return {"ts": round(float(time.time() if ts is None else ts), 3),
            "mode": "shadow", "action": (_names_only([action]) or ["?"])[0],
            "brain": _names_only(brain), "agree": bool(agree)}


def on_row(action: str, ok: bool, ts=None) -> dict:
    return {"ts": round(float(time.time() if ts is None else ts), 3),
            "mode": "on", "action": (_names_only([action]) or ["?"])[0],
            "ok": bool(ok)}


def append_row(path: str, row: dict, max_bytes: int = LOG_MAX_BYTES) -> bool:
    """Append one row; past ``max_bytes`` the file moves to ``path + '.1'``
    first. True when written. Never raises."""
    try:
        try:
            if os.path.getsize(path) >= max_bytes:
                os.replace(path, path + ".1")
        except OSError:
            pass
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        line = json.dumps(row, ensure_ascii=True, sort_keys=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except Exception:
        return False


def _valid_row(row) -> bool:
    if not isinstance(row, dict):
        return False
    ts = row.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return False
    if not isinstance(row.get("action"), str) or not _NAME_RE.match(
            row["action"]):
        return False
    if row.get("mode") == "shadow":
        return isinstance(row.get("agree"), bool)
    if row.get("mode") == "on":
        return isinstance(row.get("ok"), bool)
    return False


def read_rows(path: str) -> list:
    """Every valid row of ``path + '.1'`` then ``path``, oldest first. A
    missing file, a torn line or a malformed row is skipped. Never raises."""
    rows: list = []
    for p in (path + ".1", path):
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if _valid_row(row):
                        rows.append(row)
        except OSError:
            continue
    return rows
