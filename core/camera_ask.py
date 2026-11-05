"""core/camera_ask.py - questions about the physical world in front of the
camera (v3.0.11; CAMERA_ASK_ROUTE, LOOK_RETRY_ROUTE, CAMERA_CLAIM_CHECK).

Live 2026-11-05 14:58:03 "Jarvis, what am I holding?" got "You're holding a box
of Tic Tacs" and 14:58:31 "a smartphone", with no action run: the brain made it
up. The answer can only come from the webcam, so such a question is claimed
before the brain gets a chance to guess: the token is [ACTION: see_user, <his
words>] and the vision model's answer is what gets spoken.

Every grammar here matches the WHOLE utterance (wake word and courtesy allowed
around it), so a line that names the screen or the music - "what's on my
screen", "what is this page", "what is this error", "what's this song" - or
that merely contains the words - "what am I doing tomorrow", "how do I look up
a number" - never matches. Two strengths:

  STRICT     can only mean the physical world: "what am I holding", "what's in
             my hand", "how do I look". Always claimed.
  AMBIGUOUS  could be the screen too: "what is this", "look at this", "can you
             see this", "what colour is this". Claimed unless something else
             just ran (a page he opened, a song he played, a screen look) or
             the window in front just changed, when the old path keeps it.

A retry ("look again", "look harder", "try again", "check again") follows the
last look when one ran a moment ago: a camera look re-runs see_user with the
earlier question, a screen look stays a screen look. Any other action since
then, or no look at all, leaves the line alone.

Stdlib only, no I/O, nothing here raises.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Iterable, Optional

# The actions that look through a camera, and the ones that read the monitors.
CAMERA_LOOK_ACTIONS = frozenset({"see_user", "look_around", "recognize_face"})
SCREEN_LOOK_ACTIONS = frozenset({"see_screen", "recall_screen", "last_screen",
                                 "previous_screen", "screen_history"})

STRICT = "strict"
AMBIGUOUS = "ambiguous"

# A retry that says "look" follows the last look for this long; a bare "try
# again" / "check again" is not tied to looking, so it gets less.
LOOK_RETRY_WINDOW_S = 120.0
GENERIC_RETRY_WINDOW_S = 60.0
# "What is this?" right after another action is about that action.
AMBIGUOUS_YIELD_S = 60.0

_CONTRACTIONS = tuple((re.compile(p), r) for p, r in (
    (r"\bwhat'?s\b", "what is"),
    (r"\bi'?m\b", "i am"),
    (r"\bi'?ve\b", "i have"),
    (r"\bthat'?s\b", "that is"),
    (r"\bcolour\b", "color"),
))

_LEAD_RE = re.compile(
    r"^(?:(?:hey|hi|ok|okay|alright|all right|so|and|um+|uh+|er+|erm|well|"
    r"jarvis|sir|please|now|wait|hold on|actually|yo|dude|right|no|nope|"
    r"come on|then)\s+)+")
_TAIL_RE = re.compile(
    r"(?:\s+(?:sir|jarvis|dude|man|buddy|mate|please|then|exactly|actually|"
    r"again))+$")
_RETRY_TAIL_RE = re.compile(
    r"(?:\s+(?:sir|jarvis|dude|man|buddy|mate|please|then))+$")

_NOW = r"(?: (?:right now|now|at the moment|here|this time))?"
_OBJECT_VERB = (r"(?:holding(?: up)?|carrying|clutching|gripping|drinking|"
                r"eating|sipping|chewing)")
_BODY_WORD = (r"(?:shirt|hat|top|jacket|hoodie|tie|shoes|glasses|outfit|hair|"
              r"beard|face|necklace|watch|hands?)")
_HAND = r"(?: in my (?:left |right |other )?hands?)?"
_WEARING = r"wearing(?: today)?"
_THING = r"(?: (?:thing|object|item|one))?"
_OWNER_THING = (r"(?:i am (?:holding(?: up)?|carrying|wearing|drinking|eating)|"
                r"i have got|i have|i got)")
_SEE_LEAD = (r"(?:(?:can|could|would|will) you (?:please )?"
             r"(?:tell me |see |guess |say )?|(?:do|did) you (?:see|know) |"
             r"you see |tell me |guess |say |see |any idea |"
             r"do you have any idea |what do you think |what would you say )")
_LOOK_VERB = (r"(?:(?:can|could|would|will) you (?:please )?)?(?:please )?"
              r"(?:look at|take a look at|have a look at)")

# "what am I holding", "what the hell am I holding", "what colour shirt am I
# wearing", "what kind of drink am I holding", all with "right now" and the
# like. Not "what stock am I holding": no other noun is allowed in.
_WHAT_AM_I = (
    r"what(?: the hell| on earth| exactly)?"
    r"(?: (?:color|brand|kind|sort|type)(?: of)?(?: [a-z]+)?|"
    r" (?:shirt|hat|top|jacket|tie|shoes|drink|can|bottle|cup|glasses|thing|"
    r"object|item))? am i ")

_STRICT_RES = tuple(re.compile(p) for p in (
    _WHAT_AM_I + _OBJECT_VERB + _HAND + _NOW,
    _WHAT_AM_I + _WEARING + _NOW,
    _SEE_LEAD + r"what (?:i am|am i) " + _OBJECT_VERB + _HAND + _NOW,
    _SEE_LEAD + r"what (?:i am|am i) " + _WEARING + _NOW,
    r"what (?:do you think|would you say) (?:i am|am i) " + _OBJECT_VERB + _HAND
    + _NOW,
    r"what (?:do you think|would you say) (?:i am|am i) " + _WEARING + _NOW,
    # "doing" only bare or "right now": "what am I doing tomorrow", "what am
    # I doing wrong" and "what am I doing today" are other questions.
    r"what am i doing(?: right now| now| at the moment)?",
    r"what am i (?:pointing (?:at|to)|showing you|sitting on)",
    r"what (?:do i have|have i got|i have|i got) (?:in|on) my "
    r"(?:left |right |other )?(?:hands?|palm|lap)" + _NOW,
    r"what is (?:this |that |it )?(?:thing |object |item )?in my "
    r"(?:left |right |other )?(?:hands?|palm|lap)" + _NOW,
    r"what is (?:this|that|it)" + _THING + r" " + _OWNER_THING + _NOW,
    r"is there (?:anything|something) in my hands?",
    r"am i (?:holding|wearing) (?:anything|something)",
    r"how do i look" + _NOW + r"(?: today)?",
    r"do i look (?:ok|okay|alright|all right|good|fine|presentable|tired|"
    r"sharp|decent)" + _NOW + r"(?: today)?",
    r"how does (?:my " + _BODY_WORD + r"|this " + _BODY_WORD + r"|this) look "
    r"on me",
    r"how does my " + _BODY_WORD + r" look" + _NOW,
    _LOOK_VERB + r" what " + _OWNER_THING + _NOW,
    _LOOK_VERB + r" my " + _BODY_WORD,
    r"what color (?:is|are) my " + _BODY_WORD,
    r"what color (?:is|are) (?:this|that)" + _THING + r" " + _OWNER_THING,
))
_AMBIGUOUS_RES = tuple(re.compile(p) for p in (
    r"what is this" + _THING,
    r"what color is this" + _THING,
    r"(?:(?:can|could|do|did) )?you see this" + _THING,
    _LOOK_VERB + r" this" + _THING,
    r"what (?:do i have|have i got) here",
    r"(?:(?:can|could) you (?:see|tell(?: me)?) |tell me |do you know )"
    r"what this is",
))
# A wider test than the route: his words are about what he has or wears, even
# with more in the line ("what do you think of what I'm holding?"). Used only
# once a reply has already made such a claim (claim_validator). Stocks and
# meetings are not looks.
_MENTIONS_RE = re.compile(
    r"\b(?:(?:am i|i am) (?:holding|wearing|carrying|drinking|eating)|"
    r"in my (?:left |right |other )?hands?|how do i look)\b")
_NOT_PHYSICAL_RE = re.compile(
    r"\b(?:stocks?|shares?|positions?|bonds?|crypto|portfolio|grudge|meetings?|"
    r"calls?|conference|seminar|webinar|interview|party|event)\b")

_LOOK_RETRY_RE = re.compile(
    r"(?:(?:can|could|would|will) you (?:please )?)?(?:please )?(?:"
    r"look (?:again|harder|closer|more closely|once more|one more time|"
    r"a (?:bit|little) (?:closer|harder))"
    r"|(?:take|have|give it) (?:another|a closer|a second) look"
    r"(?: at (?:it|that|this))?"
    r"|(?:another|one more) look"
    r")")
_GENERIC_RETRY_RE = re.compile(
    r"(?:(?:can|could|would|will) you (?:please )?)?(?:please )?"
    r"(?:try|check|recheck)(?: (?:it|that|this))?"
    r"(?: again| once more| one more time)")

_WAKE_LEAD_RAW_RE = re.compile(
    r"^\W*(?:(?:hey|hi|ok|okay|alright|so|and|um+|uh+|well|jarvis|sir|please|"
    r"now|wait|actually)\b[\s,.:;!-]*)+", re.IGNORECASE)
_NOTE_RE = re.compile(r"\s*\(note: camera [^)]*\)\s*$", re.IGNORECASE)
_LOCAL_TAG_RE = re.compile(r"^\s*\[local-vision\]\s*", re.IGNORECASE)
_NOT_AN_ANSWER_RE = re.compile(
    r"^\s*(?:\(|no cameras?\b|no webcam\b|no frame\b|failed to\b|refused\b|"
    r"⚠)", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_MAX_SPOKEN_SENTENCES = 3
_MAX_SPOKEN_CHARS = 420


def _norm(text) -> str:
    try:
        t = str(text or "").lower().replace("’", "'").replace("‘", "'")
        t = re.sub(r"[^a-z0-9' ]+", " ", t)
        for rx, rep in _CONTRACTIONS:
            t = rx.sub(rep, t)
        return " ".join(t.split())
    except Exception:
        return ""


def _body(text) -> str:
    """The normalised words with the wake word and courtesy taken off."""
    t = _norm(text)
    t = _LEAD_RE.sub("", t + " ").strip()
    return _TAIL_RE.sub("", t).strip()


def ask_kind(text) -> Optional[str]:
    """STRICT or AMBIGUOUS when the whole line asks about the physical world in
    front of the camera, else None (see the module docstring)."""
    t = _body(text)
    if not t or len(t) > 120:
        return None
    if any(rx.fullmatch(t) for rx in _STRICT_RES):
        return None if _NOT_PHYSICAL_RE.search(t) else STRICT
    if any(rx.fullmatch(t) for rx in _AMBIGUOUS_RES):
        return AMBIGUOUS
    return None


def is_strict_ask(text) -> bool:
    """True when the words can only be about the physical world: what
    core.action_fit refuses a screen look for."""
    return ask_kind(text) == STRICT


def mentions_physical(text) -> bool:
    """True for a camera ask, or a longer line about what he has or wears
    ("what do you think of what I'm holding?")."""
    if ask_kind(text):
        return True
    t = _norm(text)
    return bool(_MENTIONS_RE.search(t)) and not _NOT_PHYSICAL_RE.search(t)


def retry_kind(text) -> Optional[str]:
    """"look" for "look again" / "look harder" / "take another look", "generic"
    for "try again" / "check again", None for anything else."""
    # Not _body: "again" is courtesy on an ask ("what am I holding again?") but
    # the whole point here.
    t = _LEAD_RE.sub("", _norm(text) + " ").strip()
    t = _RETRY_TAIL_RE.sub("", t).strip()
    if not t:
        return None
    if _LOOK_RETRY_RE.fullmatch(t):
        return "look"
    if _GENERIC_RETRY_RE.fullmatch(t):
        return "generic"
    return None


def question_words(text) -> str:
    """His own words for the vision model: no wake word, no brackets (a token
    argument cannot hold them), at most 180 characters."""
    s = re.sub(r"[\[\]\r\n]+", " ", str(text or ""))
    s = _WAKE_LEAD_RAW_RE.sub("", s, count=1)
    s = " ".join(s.split()).strip(" ,;:-")
    return s[:180].strip()


def token(action: str, arg: str = "") -> str:
    arg = question_words(arg) if arg else ""
    return f"[ACTION: {action}, {arg}]" if arg else f"[ACTION: {action}]"


def _rows(history: Iterable, now: float, window_s: float) -> list:
    """(name, arg) of the actions in the last ``window_s``, newest first."""
    out = []
    try:
        for row in reversed(list(history or ())):
            if not isinstance(row, dict):
                continue
            at = row.get("at")
            if not isinstance(at, (int, float)) or now - at > window_s:
                continue
            nm = str(row.get("action") or "").strip().lower()
            if nm and not nm.startswith("_"):
                out.append((nm, str(row.get("arg") or "")))
    except Exception:
        pass
    return out


def look_retry_token(text, history: Iterable = (), now: Optional[float] = None
                     ) -> Optional[str]:
    """The token a retry re-runs the last look with, when the newest action in
    its window was a look; else None. Never raises."""
    try:
        kind = retry_kind(text)
        if not kind:
            return None
        when = time.time() if now is None else float(now)
        window = (LOOK_RETRY_WINDOW_S if kind == "look"
                  else GENERIC_RETRY_WINDOW_S)
        rows = _rows(history, when, window)
        if not rows:
            return None
        name, arg = rows[0]
        if name == "look_around":
            return token("look_around")
        if name in CAMERA_LOOK_ACTIONS:
            return token("see_user", arg if name == "see_user" else "")
        if name in SCREEN_LOOK_ACTIONS:
            # A bare "try again" after a click and a look means the click: only
            # a turn of nothing but looks is a screen ask.
            if kind == "generic" and any(n not in SCREEN_LOOK_ACTIONS
                                         for n, _a in rows):
                return None
            return token("see_screen", arg)
        return None
    except Exception:
        return None


def route_token(text, history: Iterable = (), *, now: Optional[float] = None,
                ask_on: bool = True, retry_on: bool = True,
                focus_changed: Optional[Callable[[], bool]] = None
                ) -> Optional[str]:
    """The token this line is claimed with, else None. ``focus_changed`` (asked
    only for an AMBIGUOUS line): the window in front just changed, so "what is
    this?" is about it. Never raises."""
    try:
        when = time.time() if now is None else float(now)
        if retry_on:
            got = look_retry_token(text, history, when)
            if got:
                return got
        if not ask_on:
            return None
        kind = ask_kind(text)
        if not kind:
            return None
        if kind == AMBIGUOUS:
            for nm, _a in _rows(history, when, AMBIGUOUS_YIELD_S):
                if nm not in CAMERA_LOOK_ACTIONS:
                    return None
            if focus_changed is not None and focus_changed():
                return None
        return token("see_user", text)
    except Exception:
        return None


def is_camera_route_token(reply) -> bool:
    """True for the token route_token makes for a webcam look."""
    return str(reply or "").startswith(("[ACTION: see_user", "[ACTION: look_around"))


def spoken_answer(result) -> Optional[str]:
    """What the vision model said, ready to speak: no local-model tag, no
    stale-frame note, at most three sentences. None when the result is no
    answer (no camera, vision failed), so the normal failure path reports it.
    Never raises."""
    try:
        if not isinstance(result, str):
            return None
        s = _LOCAL_TAG_RE.sub("", result)
        s = _NOTE_RE.sub("", s).strip()
        if not s or _NOT_AN_ANSWER_RE.match(s):
            return None
        pieces = _SENTENCE_RE.split(" ".join(s.split()))
        out = " ".join(pieces[:_MAX_SPOKEN_SENTENCES]).strip()
        if len(out) > _MAX_SPOKEN_CHARS:
            cut = out[:_MAX_SPOKEN_CHARS]
            out = cut[:cut.rfind(" ")].rstrip(" ,;:-") + "." if " " in cut else cut
        return out or None
    except Exception:
        return None
