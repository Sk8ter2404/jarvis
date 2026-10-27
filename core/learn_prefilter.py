"""Which owner turns can teach JARVIS something -- a pure, cheap pre-filter.

THE PROBLEM (v3 plan W09, 2026-10-27 log audit): background memory extraction
(learn_from_turn / core/learn_batch.py) sent 279 local-model calls against 85
voice turns, for about two learned-fact lines. With one Ollama slot every
batch evicts the owner's cached prompt, so the next reply re-reads it (18% of
current-era turns ran a partial or full prefill: +2 to +4.5 s each).

Most turns cannot teach anything: "open Chrome", "pause the music", "thanks",
"what time is it", "what's my battery". ``worth_learning`` says so BEFORE a
turn is queued for extraction, with no model call and no I/O. It KEEPS
anything that states something about the owner or asks to be remembered:

  * "remember ...", "call me ...", "my name is ...", "don't forget ..."
  * first-person statements ("I am / I'm / I have / I like / I prefer / I
    work ...") and "my <thing> is ..." statements
  * preferences ("always", "never", "from now on") and corrections ("no, I
    said ...", "that's wrong")

and SKIPS: one-to-three-word turns, thanks / acknowledgements, questions put
to JARVIS, imperative commands ("play ...", "turn up ...", "tell Claude ..."),
polite requests ("can you ...", "I want you to ..."), and -- when an action
ran and nothing above kept the turn -- the action-only turn that is left.

The default for anything it cannot classify is KEEP (the safe direction:
learning is never lost to a clever guess). Pure stdlib; never raises; the
function takes the turn's text only, and nothing here logs it.

    decide(user_msg, ai_reply, ran_actions) -> (bool, reason)
    worth_learning(user_msg, ai_reply, ran_actions) -> bool
"""
from __future__ import annotations

import re

__all__ = ["worth_learning", "decide", "normalize_mode", "MODES"]

MODES = ("off", "shadow", "on")


def normalize_mode(value) -> str:
    """LEARN_PREFILTER as 'off' | 'shadow' | 'on'. A bool True means 'on',
    False 'off'; anything unrecognised is 'off' (the old behaviour)."""
    if value is True:
        return "on"
    if value is False or value is None:
        return "off"
    try:
        mode = str(value).strip().lower()
    except Exception:
        return "off"
    return mode if mode in MODES else "off"


# ── text helpers ─────────────────────────────────────────────────────────

_WAKE_RE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|okay|ok|yo)[\s,]+)?(?:jarvis|jarvas|service)\b"
    r"[\s,.:;!?-]*", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*;\s+|\n+")
_WORD_RE = re.compile(r"[a-z0-9']+")


def _clean(text) -> str:
    t = text if isinstance(text, str) else ""
    t = t.replace("’", "'").replace("‘", "'")
    t = _WAKE_RE.sub("", t.strip(), count=1)
    return t.strip()


def _words(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def _sentences(text: str) -> list:
    return [s.strip(" ,") for s in _SENT_SPLIT.split(text) if s and s.strip(" ,")]


# ── KEEP patterns ────────────────────────────────────────────────────────

# A turn that starts as a question about remembering ("do you remember ...")
# asks JARVIS for something; it does not hand him something to remember.
_QUESTION_LED = re.compile(
    r"^(?:what|what's|whats|when|where|who|whom|whose|why|how|how's|which|"
    r"is|isn't|are|aren't|was|were|do|does|did|can|could|will|would|should|"
    r"shall|may|might|am|has|have|had)\b", re.I)

_STRONG_KEEP = re.compile(
    r"\bremember\b|\bcall me\b|\bmy name(?:'s| is| was)\b|\bdon'?t forget\b|"
    r"\bfrom now on\b|\b(?:stop|don'?t) calling me\b|"
    r"\b(?:i|please) (?:want|need|would like) you to (?:know|remember)\b|"
    r"\bkeep in mind\b|\bnote that\b|\bfor the record\b", re.I)

_NAME_STATED = re.compile(r"\bcall me\b|\bmy name(?:'s| is| was)\b", re.I)

# A polite request to be told a fact ("can you remember that I'm vegan",
# "will you remember my wife's name is Anna") hands JARVIS something to
# keep even though it opens like a question. It needs that/this or a
# first-person word right after the verb, so "can you remember what I said"
# and "do you remember ..." stay skips.
_POLITE_REMEMBER = re.compile(
    r"^(?:can|could|will|would) you (?:please )?(?:remember|keep in mind|note)"
    r"\s+(?:(?:that|this)\b|i\b|i'm\b|i've\b|i'll\b|my\b|our\b)", re.I)

# Corrections of what JARVIS did / said / heard.
_CORRECTION = re.compile(
    r"\bi meant\b|\bi said\b|\bthat'?s (?:not|wrong|incorrect|not right)\b|"
    r"^no[,.!]?\s+(?:it'?s|that'?s|i|not|my)\b|^actually[,]?\s+(?:it|i|my|that)\b|"
    r"\bit'?s actually\b|\bnot what i (?:asked|meant|said)\b|\bwrong (?:one|app|"
    r"window|monitor|screen|song)\b", re.I)

# A sentence that STARTS with one of these states a durable fact about the
# owner (preferences and habits). These keep even a three-word turn.
_FP_PREF = re.compile(
    r"^(?:(?:and|but|so|well|yeah|yes|also|oh|um|uh|actually|honestly)[\s,]+)*"
    r"i(?:'m| am)? (?:really |also |still |just )?"
    r"(?:like|love|hate|prefer|enjoy|dislike|work|live|own|drive|study|use|"
    r"play|watch|read|collect|build|make|run|speak|attend|go to|am allergic|"
    r"can't stand|don't like|do not like|don't eat|do not eat|don't drink)\b",
    re.I)

# First-person statements that keep a turn longer than three words.
_FP_STATE = re.compile(
    r"^(?:(?:and|but|so|well|yeah|yes|also|oh|um|uh|actually|honestly)[\s,]+)*"
    r"(?:i'm|im|i am|i was|i've|ive|i have|i had|i got|i just got|i bought|"
    r"i moved|i started|i finished|i graduated|i'll be|i will be|i'd rather|"
    r"i always|i never|i usually|i tend|i often|i need to|i have to|i've got|"
    r"we have|we're|we are|we got|my)\b", re.I)

# "I have a question", "I'm back": first-person, but nothing to learn.
_FP_EMPTY = re.compile(
    r"^(?:i'm|im|i am|i've got|i have)\s+(?:a |an |one |another )?"
    r"(?:question|request|favor|problem|issue|back|here|home|ready|done|"
    r"finished|sorry|good|fine|ok|okay|alright|trying|going to (?:ask|say|"
    r"tell|need|have you)|asking|wondering|about to (?:ask|say|tell))\b",
    re.I)

# ── SKIP patterns ────────────────────────────────────────────────────────

_THANKS = re.compile(
    r"^(?:thanks?|thank you|thx|ty|cheers|much appreciated|appreciate it|"
    r"great|perfect|awesome|nice|cool|ok|okay|alright|good|got it|"
    r"never ?mind|sounds good|that works|yes|yeah|yep|no|nope|sure|please|"
    r"stop|wow|lol|hmm+|huh|oh)\b", re.I)

_POLITE_REQUEST = re.compile(
    r"^(?:please\b|can you\b|could you\b|would you\b|will you\b|can u\b|"
    r"go ahead and\b|i want you to\b|i need you to\b|i'd like you to\b|"
    r"i would like you to\b|i want it to\b|i need it to\b|i want to (?:see|"
    r"know|hear|ask)\b|let'?s\b|lets\b|how about\b|why don'?t you\b|"
    r"you (?:should|need to|have to|can)\b|try to\b|make sure\b)", re.I)

# Imperative / command verbs. The STT often hears the verb in the third
# person ("Jarvis plays X", "paused the music"), so s / ed / d are allowed.
_VERBS = (
    "open close play pause stop skip turn set show tell give find search look "
    "check put make start launch run switch mute unmute lower raise increase "
    "decrease resume next previous go bring take get send read write move "
    "minimize maximize hide dismiss cancel delete clear save screenshot "
    "capture record scan list say speak repeat ask pass talk shut restart "
    "reboot sleep wake enable disable activate deactivate toggle click "
    "press type scroll zoom focus select pick choose load download install "
    "update upgrade connect disconnect pair print queue add remove drop "
    "snap tile lock unlock dim brighten silence shuffle loop replay rewind "
    "forward fast navigate display describe explain translate calculate "
    "convert remind wait hold keep let call text email message dial ring "
    "browse watch listen cast mirror share tap swipe drag hang").split()
_IMPERATIVE = re.compile(
    r"^(?:(?:and|then|now|also|just|ok|okay|so|alright|hey)[\s,]+)*"
    r"(?:" + "|".join(sorted(map(re.escape, _VERBS), key=len, reverse=True))
    + r")(?:s|es|ed|d|ing)?\b", re.I)

# Bare command noun phrases the STT drops the verb from ("hand tracking off",
# "card mode", "volume 50", "mute", "next song").
_COMMANDISH = re.compile(
    r"\b(?:on|off|mode|volume|louder|quieter|brighter|dimmer|up|down|next|"
    r"previous|back|home page|full ?screen)\s*[.!]?$", re.I)


def _ran_names(ran_actions) -> frozenset:
    try:
        if not ran_actions:
            return frozenset()
        return frozenset(str(n) for n in ran_actions if n)
    except Exception:
        return frozenset()


def decide(user_msg, ai_reply="", ran_actions=()) -> tuple:
    """(worth_learning, reason). ``reason`` is a short fixed word -- never
    the turn's text. Never raises (any error = keep)."""
    try:
        return _decide(user_msg, ai_reply, ran_actions)
    except Exception:
        return True, "error"


def _decide(user_msg, ai_reply, ran_actions) -> tuple:
    text = _clean(user_msg)
    words = _words(text)
    if not words:
        return False, "empty"
    low = text.lower()
    sents = _sentences(text) or [text]
    question_led = bool(_QUESTION_LED.match(low))

    # 1. Strong, explicit signals keep the turn whatever else it says.
    if _STRONG_KEEP.search(low):
        # "do you remember ..." / "what's my name" ask JARVIS for something;
        # only a stated name or "call me" still counts inside a question.
        if (not question_led or _NAME_STATED.search(low)
                or _POLITE_REMEMBER.match(low)):
            return True, "explicit"
    try:
        reply = (ai_reply or "").lower()
    except Exception:
        reply = ""
    if reply and re.search(
            r"\b(?:i'?ll remember|i will remember|i'?ve noted|noted that|"
            r"i'?ll make a note|i'?ll keep that in mind)\b", reply):
        return True, "reply-remembers"
    if not question_led and _CORRECTION.search(low):
        return True, "correction"

    # 2. Durable preferences / habits start a sentence: keep, any length.
    if any(_FP_PREF.match(s) for s in sents):
        return True, "preference"

    # 3. One-to-three-word turns teach nothing.
    if len(words) <= 3:
        return False, "short"

    # 4. Thanks and acknowledgements (no stated fact left in them).
    if _THANKS.match(low) and len(words) <= 8:
        return False, "thanks"

    # 5. A first-person / possessive statement.
    if all(_FP_EMPTY.match(s) for s in sents):
        return False, "filler"
    for s in sents:
        if _FP_EMPTY.match(s):
            continue
        if _FP_STATE.match(s):
            return True, "statement"

    # 6. Polite requests and questions put to JARVIS.
    if _POLITE_REQUEST.match(low):
        return False, "request"
    if low.rstrip().endswith("?") or question_led:
        return False, "question"

    # 7. Imperative commands.
    if _IMPERATIVE.match(low) or _COMMANDISH.search(low):
        return False, "command"
    if any(_IMPERATIVE.match(s.lower()) for s in sents) and len(sents) == 1:
        return False, "command"

    # 8. An action ran and nothing above found something to learn.
    if _ran_names(ran_actions):
        return False, "action-turn"

    # 9. Anything else is a plain statement: let the extractor judge it.
    return True, "statement-default"


def worth_learning(user_msg, ai_reply="", ran_actions=()) -> bool:
    """True when a turn may teach JARVIS a fact; False for a command, thanks,
    question or one-to-three-word turn. Pure; never raises (error = True)."""
    return decide(user_msg, ai_reply, ran_actions)[0]
