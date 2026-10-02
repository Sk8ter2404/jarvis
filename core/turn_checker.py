"""core/turn_checker.py — did the local brain's turn fail in a way a retry on
Claude would fix?

WHY THIS MODULE EXISTS
======================
The local brain answers every voice turn. An escalation tier retries a turn
ONCE on Claude when the local reply failed in one of the three ways below.
This module is the DECISION half only: pure, no I/O, no LLM call, no monolith
import. The caller runs ``check_turn`` after the turn's actions ran, asks
``should_escalate``, and speaks ``ONE_MOMENT_LINE`` while Claude works.

A false escalation costs money and seconds of latency, and a turn retried for
nothing can say things twice, so PRECISION BEATS RECALL: "ok" is the default
and every kind needs positive evidence.

THE THREE FAILURE KINDS (checked in this order; the first that fires wins)
=========================================================================
  made_up_action     the reply's [ACTION: name] is not a registered action
                     and autocorrect would not resolve it
                     (command_autocorrect.autocorrect_command_choice, lexical
                     scoring, the dispatcher's own threshold and gap). An
                     action that ran but that the reply never named is the
                     runtime's autocorrect (which also uses embeddings)
                     resolving it, so that is never "made up".
  said_no_action     no action ran and none was emitted, and the reply CLAIMS
                     one ("I've set a timer", "Done, sir.", "Opening YouTube
                     now") — core/claim_validator.find_unverified_claim, the
                     rules the in-turn self-correction already uses.
  command_no_action  no action ran and none was emitted, and the owner's
                     words are a clear imperative for a registered action —
                     core/dispatcher.match_single_intent and the chain
                     resolver, the anchored rules Controlled mode executes
                     with no LLM at all. A question (claim_validator.
                     looks_like_question) is never a command.

NEVER FLAGGED
=============
  * a turn that asks the owner something — a question back, an offer
    (claim_validator.asks_owner), a confirmation request ("say 'yes' to
    proceed"): it waits on the owner, and a retry would talk over it;
  * a reply that declines or says it can't ("I'm afraid I can't do that");
  * a claim about an EARLIER turn ("did you send it?" -> "I sent it, sir.",
    "I turned them off an hour ago");
  * chit-chat, jokes and answers: no claim and no imperative;
  * words the dispatcher's media rules also read that are not clear
    commands on their own ("continue", "next", "go back"), "play" + a game
    ("play twenty questions"), and "can you X?" answered "I can".

WHY NOT THE OTHER TABLES
========================
core/prompt_router's keyword tables are tuned for RECALL ("when in doubt it
INCLUDES a section": "open" loads the app launcher for "an open question"),
so they cannot say an utterance IS a command. The monolith's preemptive-
hallucination patterns read the REPLY, not the owner, and the reply side is
claim_validator's job here. INFORMATIVE_ACTIONS only decides whether a result
is read back; here any action that ran counts.

CONFIDENCE
==========
``should_escalate`` needs ``confidence >= ESCALATE_MIN_CONFIDENCE``. "ok" is
0.0. A made-up action next to actions that DID run scores below the bar:
retrying the whole turn would run those again (a second timer, a second
email).

``Verdict.reason`` is fixed wording plus registered action names, never the
owner's or the reply's words, so it is safe to log.

Stdlib plus three repo modules, none doing I/O here (autocorrect is scored
lexically, never over its embeddings endpoint), so it is testable on the
light-deps CI runner (tests/test_turn_checker.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import command_autocorrect as _autocorrect
from core import claim_validator as _claim_validator
from core import dispatcher as _dispatcher

__all__ = [
    "COMMAND_NO_ACTION",
    "ESCALATE_MIN_CONFIDENCE",
    "KINDS",
    "MADE_UP_ACTION",
    "OK",
    "ONE_MOMENT_LINE",
    "SAID_NO_ACTION",
    "Verdict",
    "check_turn",
    "should_escalate",
]

OK = "ok"
SAID_NO_ACTION = "said_no_action"
COMMAND_NO_ACTION = "command_no_action"
MADE_UP_ACTION = "made_up_action"
KINDS = (OK, SAID_NO_ACTION, COMMAND_NO_ACTION, MADE_UP_ACTION)

# A verdict escalates only at or above this confidence.
ESCALATE_MIN_CONFIDENCE = 0.8

# Spoken while the turn is retried on Claude. One of the TTS pre-render
# OPENERS (core/tts_render_cache.py), so it plays at once.
ONE_MOMENT_LINE = "One moment, sir."

_CONF_MADE_UP = 0.95
_CONF_MADE_UP_PARTIAL = 0.6     # other actions ran: a retry would repeat them
_CONF_SAID = 0.9
_CONF_COMMAND = 0.85


@dataclass(frozen=True)
class Verdict:
    kind: str           # one of KINDS
    reason: str         # fixed wording + registry names; no transcript text
    confidence: float   # how sure the failure is; 0.0 for "ok"


# ── reply guards ─────────────────────────────────────────────────────────────
# A confirmation request without a "?" ("That one needs your confirmation,
# sir — say 'yes' to proceed") or a clarification ("tell me which room").
_CONFIRM_RE = re.compile(
    r"\bsay\s+['\"]?(?:yes|no|confirm)\b"
    r"|\bconfirm(?:ation)?\b"
    r"|\byour\s+(?:go[-\s]ahead|permission|approval)\b"
    r"|\bare\s+you\s+(?:sure|certain)\b"
    r"|\b(?:let\s+me\s+know|tell\s+me)\s+(?:which|what|whether|when|where|"
    r"how\s+(?:long|many|much))\b")

# Declines and inability ("I'm afraid I can't", "isn't connected"). Only ever
# makes a turn "ok", so a broad match costs recall, never precision.
_DECLINE_RE = re.compile(
    r"\bi\s+(?:can'?t|cannot|couldn'?t|won'?t|will\s+not|am\s+unable|"
    r"am\s+not\s+able|wasn'?t\s+able|was\s+not\s+able|don'?t\s+have|"
    r"do\s+not\s+have|have\s+no|must\s+decline|would\s+rather\s+not)\b"
    r"|\bi'd\s+rather\s+not\b"
    r"|\bi'?m\s+(?:afraid|unable|not\s+able)\b"
    r"|\b(?:unable|not\s+able)\s+to\b"
    r"|\bno\s+(?:way|means)\s+(?:to|of)\b"
    r"|\bnot\s+something\s+i\s+can\b|\bbeyond\s+(?:my|me)\b"
    r"|\b(?:isn'?t|is\s+not|aren'?t|are\s+not|not)\s+(?:connected|available|"
    r"set\s+up|configured|supported|registered|installed)\b"
    r"|\b(?:didn'?t|did\s+not|failed\s+to)\s+work\b")

# A claim about an EARLIER turn: the owner asks about one ("did you send
# it?" -> "I sent it, sir.") or the reply dates it ("I turned them off an hour
# ago", "I've already set it"). This turn cannot ground it either way.
_RECALL_ASK_RE = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|so|and|um+|uh+|well|jarvis)[,\s]+)*"
    r"(?:(?:when|what|where|why|how|which)\s+)?"
    r"(?:did|didn'?t|have|haven'?t|had)\s+you\b")
_PAST_RE = re.compile(
    r"\b(?:ago|earlier|yesterday|last\s+(?:night|time|week)|this\s+morning|"
    r"already|previously)\b")

# "Can you play some jazz?" answered "I can, sir — your library has plenty."
# is a capability question answered, not a command ignored.
_POLITE_ASK_RE = re.compile(
    r"^(?:(?:hey|ok(?:ay)?)\s+)?(?:jarvis[,\s]+)?(?:(?:so|and|um+|uh+|well)"
    r"[,\s]+)*(?:can|could)\s+you\b")
_CAN_RE = re.compile(r"\bi\s+(?:can|could|am\s+able\s+to)\b(?!'t|\s+not\b)")

# ── owner-side guards on the dispatcher's media rules ────────────────────────
# Whole utterances its rules map to an action that are not clear commands on
# their own: "continue" the story, the "next" question, "go back" a step.
_BARE_NOT_COMMANDS = frozenset({
    "continue", "next", "next one", "last", "previous", "prev", "go back",
    "silence", "skip", "skip it", "skip this",
})
# "play" + a game or an idiom: the play rule takes any object as a song.
_PLAY_NOT_MEDIA_RE = re.compile(
    r"^play\s+(?:(?:a|an|another|some|the)\s+)?(?:(?:little|quick|fun|short|"
    r"word|guessing|board|card|party)\s+)?"
    r"(?:games?|round|trivia|quiz|twenty\s+questions|20\s+questions|chess|"
    r"checkers|cards|poker|pretend|along|nice|dumb|dead|hard\s+to\s+get|"
    r"devil'?s\s+advocate|it\s+(?:cool|safe|by\s+ear)|rock\s+paper\s+scissors|"
    r"i\s+spy|would\s+you\s+rather|hide\s+and\s+seek|tag|with\s+(?:me|fire))\b")
# Trailing courtesy the anchored rules would trip on ("pause the music,
# please"); the leading kind is core.lead_fillers' job inside the dispatcher.
_TRAILING_COURTESY_RE = re.compile(
    r"(?:[\s,]+(?:please|sir|jarvis|thanks|thank\s+you))+\s*[.!?]*\s*$",
    re.IGNORECASE)


def _low(text: str) -> str:
    t = re.sub(r"\[[^\]]*\]", " ", str(text or ""))
    t = t.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t).strip().lower()


def _names(names: Optional[Iterable[str]]) -> set[str]:
    """Lower-cased action names. A leading underscore marks the runtime's
    synthetic results (_unverified_claim, _dropped_step, ...), not actions."""
    out: set[str] = set()
    for n in names or ():
        s = str(n or "").strip().lower()
        if s and not s.startswith("_"):
            out.add(s)
    return out


def _unresolved(name: str, registered: set[str]) -> bool:
    """True when autocorrect would leave ``name`` as an unknown action. When
    scoring fails the answer is "resolved": not sure means not flagged."""
    try:
        choice = _autocorrect.autocorrect_command_choice(
            name, sorted(registered), use_embeddings=False)
    except Exception:
        return False
    return choice.get("status") == "none"


def _clear_command(source: str) -> bool:
    s = " ".join(re.sub(r"[^a-z0-9' ]+", " ", source.lower()).split())
    return (bool(s) and s not in _BARE_NOT_COMMANDS
            and not _PLAY_NOT_MEDIA_RE.match(s))


def _commanded_action(user_text: str, registered: set[str]) -> Optional[str]:
    """The registered action the owner's words clearly command, else None."""
    if not registered or _claim_validator.looks_like_question(user_text):
        return None
    text = _TRAILING_COURTESY_RE.sub("", str(user_text or "").strip())
    try:
        chain = _dispatcher.command_chain_resolver(text, registered)
        steps = (chain.steps if chain is not None
                 else [_dispatcher.match_single_intent(text, registered)])
    except Exception:
        return None
    for step in steps:
        if step is not None and _clear_command(step.source):
            return step.action
    return None


# ── public API ───────────────────────────────────────────────────────────────
def check_turn(user_text: str, reply_text: str,
               actions_emitted: Iterable[str], actions_ran: Iterable[str],
               registered_actions: Iterable[str], *,
               asked_question: Optional[bool] = None) -> Verdict:
    """Judge one finished turn.

    ``user_text``          the owner's utterance.
    ``reply_text``         what the brain replied ([ACTION:] tokens may stay
                           in; bracket tags are ignored).
    ``actions_emitted``    names in the reply's [ACTION: name] tokens, as
                           written (before autocorrect).
    ``actions_ran``        names of the actions that ran this turn, including
                           any a fast path ran; an unregistered name never
                           counts as having run.
    ``registered_actions`` the live registry (the ACTIONS dict works). Empty
                           means unknown: the two checks that need it are
                           skipped.
    ``asked_question``     True when the turn asked the owner something (e.g.
                           the runtime's own "Did you mean X or Y?"); None
                           reads it from the reply.
    """
    registered = _names(registered_actions)
    emitted = _names(actions_emitted)
    ran = _names(actions_ran)
    if registered:
        ran &= registered
    asked = (_claim_validator.asks_owner(reply_text)
             if asked_question is None else bool(asked_question))
    low = _low(reply_text)
    if asked or _CONFIRM_RE.search(low):
        return Verdict(OK, "the turn asks the owner something", 0.0)
    if _DECLINE_RE.search(low):
        return Verdict(OK, "the reply declines or says it can't", 0.0)

    unknown = sorted(emitted - registered) if registered else []
    if unknown and not (ran - emitted):
        made_up = [n for n in unknown if _unresolved(n, registered)]
        if made_up:
            if ran:
                return Verdict(
                    MADE_UP_ACTION,
                    "the reply named an action that is not registered; "
                    "others ran, so a retry would repeat them",
                    _CONF_MADE_UP_PARTIAL)
            return Verdict(
                MADE_UP_ACTION,
                "the reply named an action that is not registered and "
                "autocorrect did not resolve it", _CONF_MADE_UP)
    if ran or emitted:
        return Verdict(OK, "an action was emitted or ran this turn", 0.0)

    if (not _RECALL_ASK_RE.match(_low(user_text)) and not _PAST_RE.search(low)
            and _claim_validator.find_unverified_claim(
                reply_text, ran_actions=(), user_text=user_text)):
        return Verdict(SAID_NO_ACTION,
                       "the reply claims an action but none ran", _CONF_SAID)
    action = _commanded_action(user_text, registered)
    if action and not (_POLITE_ASK_RE.match(_low(user_text))
                       and _CAN_RE.search(low)):
        return Verdict(COMMAND_NO_ACTION,
                       f"the owner asked for {action}; the reply ran no "
                       "action", _CONF_COMMAND)
    return Verdict(OK, "nothing to flag", 0.0)


def should_escalate(verdict: Verdict, *, cloud_allowed: bool,
                    already_escalated: bool,
                    needs_confirmation: bool) -> bool:
    """Retry this turn on Claude? Never twice per turn, never when the cloud
    is disallowed, never for an action that needs the owner's confirmation,
    and only for a failure at or above ESCALATE_MIN_CONFIDENCE."""
    if verdict is None or verdict.kind == OK:
        return False
    if not cloud_allowed or already_escalated or needs_confirmation:
        return False
    return verdict.confidence >= ESCALATE_MIN_CONFIDENCE
