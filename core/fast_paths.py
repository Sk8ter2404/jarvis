"""Deterministic pre-LLM answers ("fast paths").

A few questions have exactly one right answer that JARVIS already holds, and
the local model is both slow and unreliable at them (live 2026-09-29):

  * relative dates: "what's the date tomorrow", "how many days until
    Christmas", "how long until Friday" (core/date_math.py does the math);
  * "what did I just ask you" / "what was my last question": answered from
    this conversation's history with the most recent PRIOR owner utterance,
    never the current one (the LLM route answered "You just asked me what you
    had previously asked me, sir.");
  * "what's my name" / "do you know my name": answered from the configured
    owner name (core.config.USER_NAME). The LLM route sent it to the camera
    (recognize_face -> "I don't see a face right now, sir."). With no name
    configured this does NOT match, so the turn falls through to the LLM and
    nothing is ever invented. Identity/presence questions ("who am I", "who is
    this", "who am I looking at", "do you recognize me") never match: those
    stay a live camera look.

``match(text, now=..., history=..., owner_name=...)`` returns a
``FastAnswer(kind, reply)`` or None. The monolith calls it right before the LLM
dispatch (bobert_companion._run_fast_paths, gated by FAST_PATHS_ENABLED) and
speaks the reply itself. Every grammar is anchored on the whole utterance, so
commands and other domains fall through untouched. Stdlib only, no I/O, never
raises.

The recall helpers are shared with core.actions._act_session_memory_recall so
the LLM route can never recall the current utterance either.
"""
from __future__ import annotations

import re
from typing import NamedTuple, Optional

from core import date_math
from core.date_math import normalize


class FastAnswer(NamedTuple):
    kind: str    # "owner-name" | "last-utterance" | a date_math kind
    reply: str


# ── "what's my name" ───────────────────────────────────────────────────────

_NAME_RES = tuple(re.compile(p) for p in (
    r"what is my (?:first )?name",
    r"(?:do|did) you (?:know|remember) (?:my (?:first )?name|"
    r"what my (?:first )?name is)",
    r"you (?:know|remember) my (?:first )?name",
    r"(?:(?:can|could) you )?(?:tell me|say) my (?:first )?name",
    r"what am i called",
))


def is_name_question(text) -> bool:
    """True for "what's my name" style questions only. "who am I", "who is
    this", "do you recognize me" are identity looks and never match."""
    t = normalize(text)
    return any(rx.fullmatch(t) for rx in _NAME_RES)


def name_reply(owner_name) -> Optional[str]:
    """The spoken answer, or None when no name is configured (the caller then
    lets the LLM answer; a name is never invented)."""
    name = owner_name.strip() if isinstance(owner_name, str) else ""
    return f"Your name is {name}, sir." if name else None


# ── "what did I just ask you" ──────────────────────────────────────────────
# <v> names what is being recalled; it picks the reply wording.

_RECALL_RES = tuple(re.compile(p) for p in (
    r"what (?:did|do) i just (?P<v>ask|say|tell)(?: to)?(?: you| jarvis)?"
    r"(?: to do)?",
    r"what did i (?P<v>ask|say|tell)(?: you| to you)? (?:just now|"
    r"a (?:moment|second|minute|sec) ago|last|previously)",
    r"what (?:was|is) (?:my|the) (?:last|previous|most recent|prior) "
    r"(?P<v>question|request|command)(?: i asked(?: you)?| to you)?",
    r"what was the last (?:thing|question) i (?P<v>asked|said)"
    r"(?: you| to you)?",
    r"what was i just (?P<v>asking|saying)(?: you| to you)?",
    r"(?:(?:can|could) you )?(?:repeat|remind me(?: of)?|tell me) "
    r"(?:my|the) (?:last|previous) (?P<v>question|request)",
    r"(?:(?:can|could) you )?(?:repeat|remind me|tell me) what i just "
    r"(?P<v>asked|said)(?: you| to you)?",
    r"do you (?:remember|know) (?:what i just (?P<v>asked|said)"
    r"(?: you| to you)?|(?:my|the) (?:last|previous) (?P<v2>question|request))",
))
_VERBS = {"ask": "ask", "asked": "ask", "asking": "ask",
          "say": "say", "said": "say", "saying": "say", "tell": "say",
          "question": "question", "request": "request",
          "command": "command"}

# Looser detector for the ACTION path (the LLM hands session_memory_recall the
# question "verbatim", which may be a paraphrase): a current-conversation
# recall mention, with no time reference that points at an EARLIER session.
_LOOSE_RECALL_RE = re.compile(
    r"\bjust (?:ask|asked|asking|say|said|saying|tell|told)\b"
    r"|\b(?:last|previous|prior|most recent) (?:question|request|command|"
    r"message)\b"
    r"|\blast thing i (?:asked|said)\b"
    r"|\b(?:ask|asked|say|said)(?: you| to you)? (?:just now|"
    r"a (?:moment|second|minute|sec) ago)\b")
_PAST_SESSION_RE = re.compile(
    r"\b(?:yesterday|last (?:night|week|month|year|time|session)|"
    r"this (?:morning|afternoon|evening|week)|earlier today|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d+ (?:days?|weeks?|hours?) ago)\b")


def recall_verb(text) -> Optional[str]:
    """"ask" / "say" / "question" / "request" / "command" when ``text`` is a
    "what did I just ask/say" question (strict, whole-utterance), else None."""
    t = normalize(text)
    for rx in _RECALL_RES:
        m = rx.fullmatch(t)
        if m:
            v = m.groupdict().get("v") or m.groupdict().get("v2") or "ask"
            return _VERBS.get(v, "ask")
    return None


def is_last_utterance_question(text, *, loose: bool = False) -> bool:
    """True when ``text`` asks for the owner's previous utterance in THIS
    conversation. ``loose`` also accepts a paraphrase that merely mentions
    "just asked" / "last question" (the action path), but never one with an
    earlier-session time reference ("what did I ask you yesterday")."""
    if recall_verb(text):
        return True
    if not loose:
        return False
    t = normalize(text)
    return bool(_LOOSE_RECALL_RE.search(t)) and not _PAST_SESSION_RE.search(t)


_WAKE_LEAD_RE = re.compile(r"^(?:(?:hey|ok|okay)\s+)?jarvis\b[\s,.:;!-]*",
                           re.IGNORECASE)


def _clean_utterance(s: str) -> str:
    s = " ".join(s.split())
    s = _WAKE_LEAD_RE.sub("", s).strip() or s
    s = s.rstrip(" ?!.,;:")
    if len(s) > 200:
        s = s[:197].rstrip() + "..."
    return s


def prior_owner_utterance(history, *, skip_newest: bool = False
                          ) -> Optional[str]:
    """The most recent PRIOR owner utterance in ``history`` (a
    conversation_history-shaped list), cleaned for speech, or None.

    Never the current utterance:
      * ``skip_newest`` drops the newest user entry — pass it when the current
        turn is already recorded (the LLM path appends the user message before
        any action runs; see recall_turn_recorded);
      * an entry that is itself a "what did I just ask" question is skipped,
        so asking twice recalls the same real question instead of the
        previous recall question.
    """
    users = []
    for m in history or ():
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                users.append(c)
    if skip_newest and users:
        users.pop()
    for c in reversed(users):
        if is_last_utterance_question(c, loose=True):
            continue
        cleaned = _clean_utterance(c)
        if cleaned:
            return cleaned
    return None


def recall_turn_recorded(history, action: str = "session_memory_recall"
                         ) -> bool:
    """True when the newest user entry in ``history`` IS the current turn:
    an assistant entry after it carries the ``action`` token, i.e. the LLM
    path appended this turn's user message and then emitted the action.
    False on the pre-LLM paths (chain resolver, controlled mode), where the
    current utterance is not in the history yet."""
    tagged = False
    for m in reversed(list(history or ())):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            return tagged
        if role == "assistant" and action in str(m.get("content") or ""):
            tagged = True
    return False


_FOUND = {
    "ask": 'You asked me: "{u}", sir.',
    "say": 'You said: "{u}", sir.',
    "question": 'Your last question was: "{u}", sir.',
    "request": 'Your last request was: "{u}", sir.',
    "command": 'Your last command was: "{u}", sir.',
}
_NONE = {
    "ask": "You haven't asked me anything else in this conversation yet, sir.",
    "say": ("You haven't said anything else to me in this conversation yet, "
            "sir."),
    "question": "There's no earlier question from you in this conversation, "
                "sir.",
    "request": "There's no earlier request from you in this conversation, "
               "sir.",
    "command": "There's no earlier command from you in this conversation, "
               "sir.",
}


def _guess_verb(text) -> str:
    t = normalize(text)
    if re.search(r"\b(?:say|said|saying|tell|told)\b", t):
        return "say"
    for noun in ("question", "request", "command"):
        if re.search(rf"\b{noun}\b", t):
            return noun
    return "ask"


def last_utterance_reply(text, history, *, skip_newest: bool = False) -> str:
    """The spoken answer to "what did I just ask/say" — the prior utterance,
    or an honest "nothing earlier" line. Never the current utterance."""
    verb = recall_verb(text) or _guess_verb(text)
    prior = prior_owner_utterance(history, skip_newest=skip_newest)
    if prior is None:
        return _NONE[verb]
    return _FOUND[verb].format(u=prior)


# ── the one entry point ────────────────────────────────────────────────────

def match(text, *, now=None, history=(), owner_name="") -> Optional[FastAnswer]:
    """A deterministic answer for ``text``, or None to let the LLM answer.

    ``now`` is the local datetime the date questions are answered from (None
    skips them), ``history`` is conversation_history WITHOUT the current turn,
    ``owner_name`` the configured USER_NAME ("" = unknown). Never raises."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        if is_name_question(text):
            reply = name_reply(owner_name)
            return FastAnswer("owner-name", reply) if reply else None
        if recall_verb(text):
            return FastAnswer("last-utterance",
                              last_utterance_reply(text, history))
        if now is not None:
            got = date_math.answer(text, now)
            if got is not None:
                return FastAnswer(got.kind, got.reply)
    except Exception:
        return None
    return None
