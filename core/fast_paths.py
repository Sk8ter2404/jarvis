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
    nothing is ever invented.
  * "who am I" / "do you know who I am" (v2.0.148): the same configured name
    ("You're Alex, sir."). Live v2.0.140 it ran recognize_face and answered
    "I don't see a face right now, sir." These were camera looks since v1.55,
    for a real reason: the owner stepped out of frame, asked "who am I" and
    got his name back, i.e. JARVIS claimed to SEE someone it did not. That
    reason is kept, not dropped: the reply names who JARVIS is set up for and
    never claims a sighting, and every presence / recognition phrasing ("who
    is this", "who's here", "who am I looking at", "do you recognize me",
    "can you see me") still never matches, so it stays a live camera look
    (and the prompts' camera rule still governs it). No name configured =
    no match, as above.
  * "what was the first thing I asked you (today / in this conversation)",
    "what did I ask you first" (v2.0.148): the first owner utterance of THIS
    process session. conversation_history cannot answer it (it is trimmed
    from the front, and a blue-green handoff seeds it with the previous
    process's tail), so the monolith keeps the session's opening utterances
    itself (bobert_companion._session_opening_turns) and passes them in.
  * "what time is it in London" (v2.0.148): core/world_clock.py. Live
    v2.0.140 get_time (the LOCAL clock) was voiced as "It is 10:17 PM in
    London, sir." when London was at 4:17 AM.
  * spoken arithmetic, "what's 12 times 7" / "144 divided by 12" / "2 to the
    power of 10" (2026-10-01): core/spoken_math.py, evaluated exactly. The
    09-05 live diagnostic found operator words never reached the calculator
    (the local model answered from its head). Only a WHOLE-utterance
    expression with a number on both sides of each operator matches; "what
    times does the store open" falls through untouched.

Recall never returns a recall question itself or a bare wake phrase
("Jarvis", "hey Jarvis, wake up"): both are skipped (_skip_for_recall).

``match(text, now=..., history=..., owner_name=..., session_turns=...)``
returns a ``FastAnswer(kind, reply)`` or None. The monolith calls it right
before the LLM dispatch (bobert_companion._run_fast_paths, gated by
FAST_PATHS_ENABLED) and speaks the reply itself. Every grammar is anchored on
the whole utterance, so commands and other domains fall through untouched.
Stdlib only, no I/O, never raises.

The recall helpers are shared with core.actions._act_session_memory_recall so
the LLM route can never recall the current utterance either.
"""
from __future__ import annotations

import re
from typing import NamedTuple, Optional

from core import date_math, spoken_math, wake_prefix, world_clock
from core.date_math import normalize


class FastAnswer(NamedTuple):
    kind: str    # "owner-name" | "owner-identity" | "last-utterance" |
    #              "first-utterance" | "world-clock" | "arithmetic" |
    #              a date_math kind
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
    """True for "what's my name" style questions only. "who am I" is
    is_identity_question; "who is this", "do you recognize me" are camera
    looks and never match either."""
    t = normalize(text)
    return any(rx.fullmatch(t) for rx in _NAME_RES)


def name_reply(owner_name) -> Optional[str]:
    """The spoken answer, or None when no name is configured (the caller then
    lets the LLM answer; a name is never invented)."""
    name = owner_name.strip() if isinstance(owner_name, str) else ""
    return f"Your name is {name}, sir." if name else None


# ── "who am I" ─────────────────────────────────────────────────────────────
# Whole-utterance only. "who am I looking at / talking to / speaking with",
# "who is this", "who's here", "do you recognize me", "can you see me" are
# presence or recognition questions and stay a live camera look.

_IDENTITY_RES = tuple(re.compile(p) for p in (
    r"who am i",
    r"(?:(?:can|could) you )?(?:tell me|remind me) who i am",
))
_KNOW_IDENTITY_RES = tuple(re.compile(p) for p in (
    r"(?:do |did )?you (?:know|remember) who i am",
))


def is_identity_question(text) -> bool:
    """True for "who am I" / "do you know who I am" (owner identity from the
    configured name, never a camera look)."""
    t = normalize(text)
    return any(rx.fullmatch(t) for rx in _IDENTITY_RES + _KNOW_IDENTITY_RES)


def identity_reply(text, owner_name) -> Optional[str]:
    """ "You're Alex, sir." ("Of course, sir. You're Alex." to "do you know
    who I am"), or None when no name is configured. Names who JARVIS is set
    up for; never claims to see anyone."""
    name = owner_name.strip() if isinstance(owner_name, str) else ""
    if not name:
        return None
    t = normalize(text)
    if any(rx.fullmatch(t) for rx in _KNOW_IDENTITY_RES):
        return f"Of course, sir. You're {name}."
    return f"You're {name}, sir."


# ── "what did I just ask you" ──────────────────────────────────────────────
# <v> names what is being recalled; it picks the reply wording. "what'd"
# normalises to "whatd" (date_math.normalize drops the apostrophe).

_WHAT_DID = r"(?:what did|whatd)"
_RECALL_RES = tuple(re.compile(p) for p in (
    r"(?:what (?:did|do)|whatd) i just (?P<v>ask|say|tell)(?: to)?"
    r"(?: you| jarvis)?(?: to do)?",
    _WHAT_DID + r" i (?P<v>ask|say|tell)(?: you| to you)? (?:just now|"
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
          "told": "say",
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


# "what was the first thing I asked you (today / in this conversation)".
# <v> names what is being recalled, as in _RECALL_RES.
_SCOPE = (r"(?: (?:today|tonight|so far|(?:in )?(?:this|our) "
          r"(?:conversation|session|chat)|since you (?:started|started up|"
          r"came up|came online|booted|booted up|woke up)))?")
_YOU = r"(?: you| to you| of you)?"
_FIRST_RES = tuple(re.compile(p + _SCOPE) for p in (
    r"what (?:was|is) the (?:very )?first (?:thing|question) i "
    r"(?P<v>asked|said)" + _YOU,
    r"what (?:was|is) the (?:very )?first thing i (?P<v>told) you",
    r"what (?:was|is) my (?:very )?first (?P<v>question|request|command)",
    _WHAT_DID + r" i (?P<v>ask|say|tell)" + _YOU + r" first",
    _WHAT_DID + r" i first (?P<v>ask|say|tell)" + _YOU,
    r"(?:(?:can|could) you )?(?:tell me|remind me(?: of)?|repeat) "
    r"(?:what i (?P<v>asked|said)" + _YOU + r" first|the (?:very )?first "
    r"thing i (?P<v2>asked|said)" + _YOU + r"|my (?:very )?first "
    r"(?P<v3>question|request))",
    r"do you (?:remember|know) (?:what i (?P<v>asked|said)" + _YOU +
    r" first|the (?:very )?first thing i (?P<v2>asked|said)" + _YOU +
    r"|my (?:very )?first (?P<v3>question|request))",
))
# The action path's paraphrase detector (the LLM's argument): a "first thing
# I asked" mention with no earlier-session time reference. It needs the
# ASKING, BY THE OWNER, in every form: "the first thing the user asked", "his
# first question", "what I asked you first", "what did the user say first".
# A bare "first thing" is not enough — "what was the first thing I worked on
# today" / "the first thing we did today" are work-history questions for the
# session index, not the opening utterance — and neither is someone else
# asking: "the first thing you said to me today" (JARVIS), "what did my mom
# say first", "what the doctor said first", "the first message they sent".
# "he" / "they" count only when the argument itself names the user as whom
# they stand for ("user wants the first thing they asked").
_OWNER = r"(?:i|the user|user|the owner|owner|sir)"
_OWNER_PRONOUN = r"(?:he|they)"


def _loose_first_re(asker: str):
    return re.compile(
        r"\b(?:first|earliest|opening) (?:thing|question|request|command|"
        rf"message)s? (?:that )?{asker}(?: had| has| have)? (?:ever |just )?"
        r"(?:asked|said|told|gave|sent|ask|say|tell)\b"
        rf"|\b{asker} (?:ask|asked|say|said|tell|told)"
        r"(?: you| to you| jarvis)? first\b"
        rf"|\b{asker} first (?:ask|asked|say|said|tell|told)\b")


_LOOSE_FIRST_RE = re.compile(
    _loose_first_re(_OWNER).pattern
    + r"|\b(?:my|his|users?|owners?|sirs?) (?:very )?(?:first|earliest|"
    r"opening) (?:question|request|command)\b")
_LOOSE_FIRST_PRONOUN_RE = _loose_first_re(_OWNER_PRONOUN)
_USER_NAMED_RE = re.compile(r"\b(?:users?|owners?)\b")


def first_recall_verb(text) -> Optional[str]:
    """"ask" / "say" / "question" / "request" / "command" when ``text`` is a
    "what was the first thing I asked" question (strict, whole-utterance),
    else None."""
    t = normalize(text)
    for rx in _FIRST_RES:
        m = rx.fullmatch(t)
        if m:
            g = m.groupdict()
            v = g.get("v") or g.get("v2") or g.get("v3") or "ask"
            return _VERBS.get(v, "ask")
    return None


def is_first_utterance_question(text, *, loose: bool = False) -> bool:
    """True when ``text`` asks for the owner's FIRST utterance of this
    session. ``loose`` also accepts a paraphrase that merely mentions "the
    first thing I asked" (the action path), but never one with an earlier-
    session time reference ("the first thing I asked you yesterday")."""
    if first_recall_verb(text):
        return True
    if not loose:
        return False
    t = normalize(text)
    asked = bool(_LOOSE_FIRST_RE.search(t)) or bool(
        _LOOSE_FIRST_PRONOUN_RE.search(t) and _USER_NAMED_RE.search(t))
    return asked and not _PAST_SESSION_RE.search(t)


# A bare wake / attention phrase is not something he asked ("Jarvis", "hey
# Jarvis", "Jarvis, wake up", "are you there"): recall skips it.
_WAKE_ONLY_RE = re.compile(
    r"(?:(?:hey|hi|hello|ok|okay|yo|oh) )*"
    r"(?:jarvis(?: (?:wake up|are you (?:there|awake|up|listening)|"
    r"you there))?|wake up(?: jarvis)?|are you (?:there|awake|listening)"
    r"(?: jarvis)?)")


def is_wake_only(text) -> bool:
    """True when ``text`` is only a wake / attention phrase."""
    if not isinstance(text, str):
        return False
    t = " ".join(re.sub(r"[^a-z' ]+", " ", text.lower()).split())
    if t and _WAKE_ONLY_RE.fullmatch(t):
        return True
    # A wake behind lead interjections ("Um, Jarvis.", "So Jarvis, are you
    # there?"): the wake-word gate admits it (core.wake_prefix, word 1-3), so
    # it is a wake here too when nothing but a wake phrase follows the name.
    if wake_prefix.has_wake_prefix(text):
        rest = wake_prefix.strip_wake_lead(text)
        r = " ".join(re.sub(r"[^a-z' ]+", " ", rest.lower()).split())
        return not r or bool(_WAKE_ONLY_RE.fullmatch(r))
    return False


# A STORED owner utterance that was itself a recall question, for recall to
# skip. Deliberately tighter than the loose ACTION-argument detectors above:
# those see the LLM's paraphrase of the question being asked right now,
# while these see every real thing he said. It needs a recall lead AND the
# owner as the one who asked / said, so "what's the first thing on my
# calendar today", "read me the first message in my inbox", "remind me to
# call mom first thing in the morning", "read me my last message" and "what
# did the caller just say" are real requests and ARE recalled (review F1 /
# F2). The lead also takes "what'd" (normalised "whatd"), a yes/no "did I
# ..." and "go back to ...": "what'd I just ask you", "did I just ask you
# something" and "go back to my last question" are recall questions (the
# base's loose skip caught them; the second review found them recalled
# verbatim), while "cancel my last command" still has no lead and is a real
# request.
_RECALL_LEAD_RE = re.compile(
    r"\b(?:what|whatd|which|remind|tell me|repeat|remember|recall|did i|"
    r"go back|back to)\b")
_SELF_RECALL_RE = re.compile(
    r"\b(?:first|last|previous|prior|most recent|earliest|opening) "
    r"(?:thing|question|request|command)s? (?:that )?i (?:had |have )?"
    r"(?:ever |just )?(?:asked|said|told|gave|ask|say|tell)\b"
    r"|\bi (?:just|first|last) (?:ask|asked|say|said|tell|told)\b"
    r"|\bi (?:ask|asked|say|said|tell|told)(?: you| to you| jarvis)? "
    r"(?:first|last|just now|earlier|before (?:this|that)|"
    r"a (?:moment|second|minute|sec|while) ago|at the (?:start|beginning)|"
    r"when you (?:started|started up|booted|booted up|came online|came up|"
    r"woke up))\b"
    r"|\bmy (?:very )?(?:first|last|previous|prior|most recent|earliest|"
    r"opening) (?:question|request|command)\b"
    r"|\bwhat (?:was|were) i (?:just )?(?:asking|saying)\b")


def is_stored_recall_question(text) -> bool:
    """True when a STORED owner utterance was itself a recall question ("what
    did I just ask you", "what was the first thing I asked", "what's the
    earliest thing I asked you today"): recall skips it. Never an ordinary
    request that merely mentions "first thing" / "last message"."""
    if recall_verb(text) or first_recall_verb(text):
        return True
    t = normalize(text)
    return bool(_RECALL_LEAD_RE.search(t) and _SELF_RECALL_RE.search(t)
                and not _PAST_SESSION_RE.search(t))


def _skip_for_recall(text) -> bool:
    """An owner utterance recall never returns: a recall question (last OR
    first, see is_stored_recall_question) or a bare wake phrase. Stored
    utterances are judged by the tight detector, never the loose action-
    argument ones (those matched "the first thing on my calendar")."""
    return is_stored_recall_question(text) or is_wake_only(text)


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


def _clean_utterance(s: str) -> str:
    s = " ".join(s.split())
    # The lead fillers + wake word off the front: core.wake_prefix, the same
    # rule the wake-word gate admits a turn on (this module used to keep its
    # own regex that took only "Jarvis" / "hey|ok|okay Jarvis").
    s = wake_prefix.strip_wake_lead(s).strip() or s
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
      * an entry that is itself a recall question ("what did I just ask",
        "what was the first thing I asked") is skipped, so asking twice
        recalls the same real question instead of the previous recall
        question; so is a bare wake phrase ("Jarvis", "hey Jarvis, wake up").
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
        if _skip_for_recall(c):
            continue
        cleaned = _clean_utterance(c)
        if cleaned:
            return cleaned
    return None


def first_owner_utterance(session_turns) -> Optional[str]:
    """The FIRST owner utterance of this session, cleaned for speech, or
    None. ``session_turns`` is the monolith's record of the session's opening
    owner utterances, oldest first (bobert_companion._session_opening_turns,
    never conversation_history: that is trimmed and may hold a previous
    process's tail). Recall questions and bare wake phrases are skipped, the
    same rule as prior_owner_utterance."""
    for c in session_turns if isinstance(session_turns, (list, tuple)) else ():
        if not isinstance(c, str) or not c.strip() or _skip_for_recall(c):
            continue
        cleaned = _clean_utterance(c)
        if cleaned:
            return cleaned
    return None


_FIRST_FOUND = {
    "ask": 'The first thing you asked me this session was: "{u}", sir.',
    "say": 'The first thing you said to me this session was: "{u}", sir.',
    "question": 'Your first question this session was: "{u}", sir.',
    "request": 'Your first request this session was: "{u}", sir.',
    "command": 'Your first command this session was: "{u}", sir.',
}
_FIRST_NONE = {
    "ask": "You haven't asked me anything else this session yet, sir.",
    "say": "You haven't said anything else to me this session yet, sir.",
    "question": "There's no earlier question from you this session, sir.",
    "request": "There's no earlier request from you this session, sir.",
    "command": "There's no earlier command from you this session, sir.",
}
# The start of the session is gone: "forget the last hour" / "reset memory"
# purged the recorded first utterance, or a blue-green handoff arrived
# without a record of it while the conversation shows he DID say something
# (the monolith latches this, _session_opening_lost, for the rest of the
# session: every later utterance is NOT the first thing he asked). Or the
# record holds nothing real, yet the history shows an earlier owner turn.
# Never claim nothing was asked, never name a later turn as the first, and
# never recite from the history either — it is only a tail (not the start),
# and after a forget reciting it would undo the forget.
_FIRST_UNKNOWN = ("I no longer have the start of this session on record, "
                  "sir.")


def first_utterance_reply(text, session_turns, history=None, *,
                          skip_newest: bool = False,
                          start_lost: bool = False) -> str:
    """The spoken answer to "what was the first thing I asked" — the first
    owner utterance of this session, or an honest line when there is none:
    "nothing earlier", or, when ``start_lost`` (the monolith's latch: a
    forget / reset purged the start, or a handoff came without it) or when
    ``history`` (conversation_history; the current turn skipped as in
    prior_owner_utterance) shows an earlier owner utterance the record does
    not hold, that the start is not on record."""
    if start_lost:
        return _FIRST_UNKNOWN
    verb = first_recall_verb(text) or _guess_verb(text)
    first = first_owner_utterance(session_turns)
    if first is None:
        if history and prior_owner_utterance(
                history, skip_newest=skip_newest) is not None:
            return _FIRST_UNKNOWN
        return _FIRST_NONE[verb]
    return _FIRST_FOUND[verb].format(u=first)


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

# ── "are you ok" / "run a system check" (2026-10-01) ──────────────────────
# The 09-05 live diagnostic: both got a pass from the model's head ("Quite
# right, sir. Always operational.") or an off-topic CPU readout. The answer
# needs the LIVE self-diagnostic, so match() never answers these itself — the
# monolith's _run_self_check_shortcut runs the action and speaks its summary.
# Whole-utterance only: "are you ok with that" is a conversation, and the
# liveness questions ("are you there", "can you hear me") keep their own
# instant answer.
_OKAY = r"(?:ok|okay|alright|all right)"
_SELF_CHECK_RES = tuple(re.compile(p) for p in (
    rf"^(?:are|r) (?:you|u)(?: doing| feeling)? {_OKAY}$",
    rf"^(?:you|u) {_OKAY}$",
    rf"^is everything {_OKAY}(?: with you)?$",
    r"^(?:(?:please )?(?:run|do|perform|start|give me)(?: me)?"
    r"(?: a| an| the| your)?(?: quick| full| complete)? )?"
    r"(?:system|systems|self|health) ?(?:check|test|diagnostic|diagnostics)"
    r"(?: on yourself)?$",
    r"^(?:please )?(?:run|do|perform|start)(?: a| an| the| your)?"
    r"(?: full| complete)? (?:self )?diagnostics?$",
    r"^check yourself$",
))


def is_self_check_request(text) -> bool:
    """True for a whole-utterance self-check request: "are you ok", "run a
    system check", "run a diagnostic", "check yourself". Never raises."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        t = normalize(text).replace("-", " ")
        t = re.sub(r"\s+", " ", t).strip()
        return any(rx.match(t) for rx in _SELF_CHECK_RES)
    except Exception:
        return False


# ── "what timers do I have" (2026-10-01) ───────────────────────────────────
# "list_timers can make things up" (09-05 live diagnostic): the local model
# answered timer questions in its own words. The answer is the timer store's,
# so a whole-utterance listing question never reaches the model — the
# monolith's _run_timer_list_shortcut runs list_timers and speaks its line.
# Setting / cancelling a timer never matches.
_TIMERS_N = r"(?:timers?|reminders?|countdowns?)"
_TIMER_STATE_W = r"(?:running|set|going|active|pending|on)"
_TIMER_LIST_RES = tuple(re.compile(p) for p in (
    rf"^(?:can you |could you |would you )?(?:list|show|read|tell me|give me|"
    rf"check|read out|read me)(?: me)?(?: all)?(?: of)? (?:my |the |our |any )?"
    rf"(?:active |running |current |pending )?{_TIMERS_N}(?: {_TIMER_STATE_W})?$",
    rf"^(?:what|which) {_TIMERS_N} (?:do i have|have i got|are "
    rf"{_TIMER_STATE_W}|is {_TIMER_STATE_W}|do i have {_TIMER_STATE_W})$",
    rf"^(?:do i have|have i got|are there|is there|got) (?:any|a) "
    rf"(?:active |running |pending )?{_TIMERS_N}(?: {_TIMER_STATE_W})?$",
    rf"^any (?:active |running |pending )?{_TIMERS_N}(?: {_TIMER_STATE_W})?$",
    r"^how (?:much time|long) (?:is |do i have )?(?:left|remaining) on "
    r"(?:my|the) (?:\w+ )?timers?$",
    r"^what is left on (?:my|the) (?:\w+ )?timers?$",
    r"^(?:when|what time) (?:does|will|is) (?:my|the) (?:\w+ )?timer "
    r"(?:go off|going off|done|due)$",
))


def is_timer_list_request(text) -> bool:
    """True for a whole-utterance "what timers / reminders do I have"
    question. Never raises."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        t = re.sub(r"\s+", " ", normalize(text)).strip()
        return any(rx.match(t) for rx in _TIMER_LIST_RES)
    except Exception:
        return False


def match(text, *, now=None, history=(), owner_name="",
          session_turns=None, session_start_lost=False
          ) -> Optional[FastAnswer]:
    """A deterministic answer for ``text``, or None to let the LLM answer.

    ``now`` is the local datetime the date and world-clock questions are
    answered from (None skips them; a naive one is this machine's local
    time), ``history`` is conversation_history WITHOUT the current turn,
    ``owner_name`` the configured USER_NAME ("" = unknown), ``session_turns``
    this session's opening owner utterances, oldest first (None = unknown:
    "what was the first thing I asked" then falls through),
    ``session_start_lost`` True when the start of the session is no longer
    on record (see first_utterance_reply). Never raises."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        if is_name_question(text):
            reply = name_reply(owner_name)
            return FastAnswer("owner-name", reply) if reply else None
        if is_identity_question(text):
            reply = identity_reply(text, owner_name)
            return FastAnswer("owner-identity", reply) if reply else None
        if recall_verb(text):
            return FastAnswer("last-utterance",
                              last_utterance_reply(text, history))
        if first_recall_verb(text):
            if not isinstance(session_turns, (list, tuple)):
                return None
            return FastAnswer("first-utterance",
                              first_utterance_reply(
                                  text, session_turns, history,
                                  start_lost=session_start_lost is True))
        if now is not None:
            clock = world_clock.answer(text, now)
            if clock is not None:
                return FastAnswer(clock.kind, clock.reply)
            got = date_math.answer(text, now)
            if got is not None:
                return FastAnswer(got.kind, got.reply)
        # After the date grammars (a date question keeps its own answer);
        # needs no clock.
        calc = spoken_math.answer(text)
        if calc is not None:
            return FastAnswer("arithmetic", calc.reply)
    except Exception:
        return None
    return None
