"""core/claim_validator.py — is a reply CLAIMING an action nothing ran?

WHY THIS MODULE EXISTS
======================
JARVIS used to say it had done things it hadn't ("Restarting now, sir." with
no restart action emitted). ``parse_and_run_actions`` therefore runs a
reactive check on every reply that carries no ``[ACTION: ...]`` token: when the
prose claims execution, a synthetic ``_unverified_claim`` result is appended
and the follow-up loop re-prompts the LLM to emit the real token or admit it
can't. That re-prompt is a full extra LLM round plus extra speech, so a false
positive is expensive and audible.

The original check was a bare SUBSTRING match against a phrase list
("moving ", "opening ", "on it, sir", ...). Live v2.0.115 (2026-09-29) showed
the cost:

  * "the moon is actually moving about 1.5 inches away from us every year"
    matched "moving " — a third-party fact — and 11 s later JARVIS apologised
    that it "can't actually move the moon".
  * "On it, sir. You asked about <x>." answering "what did I just ask you"
    matched "on it, sir"; the forced second round produced a second, WRONG
    answer, so the owner heard two answers.

THE RULES (only first-person claims of JARVIS acting count)
===========================================================
A reply with no ``[ACTION:]`` token is flagged when it contains:

  1. A FIRST-PERSON action claim, matched with word boundaries:
       * progressive  — "I'm opening ...", "I am moving it"
       * perfect/past — "I've turned off the lights", "I have sent it",
                        "I just opened ..."
       * future       — "I'll move it", "I will restart", "I'm going to play",
                        "Let me take a look"
       * subjectless narration — a clause that STARTS with the action gerund
         or participle, optionally after courtesy lead-ins: "Restarting now,
         sir.", "Very good, sir, opening Spotify.", "Moving it now, sir.",
         "Sent, sir.", "Queued three songs." A later comma-separated part
         counts only with a narration signal ("Excellent choice, sir,
         playing it now.").
     A gerund that is not clause-initial ("the moon is moving ...") is never
     a claim. A clause-initial gerund used as a noun subject ("Moving house is
     stressful", "Playing chess can improve memory") or as a participial
     opener ("Moving about 4 cm a year, the moon ...") is not a claim either.
     Questions and offers ("Shall I open it?", "I'll open it if you'd like")
     are never claims. Idioms are excluded ("moving on", "playing devil's
     advocate", "switching gears", "closing thoughts").
  2. An execution ACKNOWLEDGEMENT with nothing else of substance — a bare
     "On it, sir." / "Right away, sir." / "Done, sir." — still a claim.
  3. An acknowledgement followed by other content is a claim UNLESS the
     owner's turn was a question and the content is a substantive answer with
     no promise of a future or in-progress action ("On it, sir. You asked about
     X." answering "what did I just ask you" is an answer with a filler
     preface; "On it, sir. The lights are off." answering "turn off the
     lights" is still flagged).

GROUNDING (follow-up rounds)
============================
In a follow-up round the reply summarises results that already came back
this turn, so "Playing X by Y, sir." after ``play_music`` ran, or "Done, sir."
after any action succeeded, is grounded, not hallucinated. The caller passes
the names of the actions that ran successfully this turn (``ran_actions``); a
claim is grounded when an action from the SAME verb family ran (a claim to
"open" something is not grounded by ``get_time``). An acknowledgement followed
by a summary, or a bare completion ack ("Done, sir."), is grounded by any
action that succeeded this turn. A bare pending acknowledgement ("On it,
sir.") is never grounded — it promises something new.

Pure: no I/O, no monolith import, stdlib ``re`` only, so it is testable on the
light-deps CI runner (tests/test_claim_validator.py).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

__all__ = [
    "find_unverified_claim",
    "is_progress_only",
    "looks_like_question",
    "strip_ack_preface",
]

# ── verb families ────────────────────────────────────────────────────────────
# Each family: gerund / participle / base regex fragments (lower-case text,
# straight apostrophes) plus the action-name tokens that GROUND a claim from
# it in a follow-up round. Negative look-aheads drop the common idioms.
_TURN_OBJ = (r"(?:(?:it|them|that|this|those|these|everything|all\s+(?:of\s+)?"
             r"(?:the\s+)?\w+|(?:the|your|my)\s+\w+(?:\s+\w+)?)\s+)?")
_TIMER_OBJ = r"(?:a|an|the|your)\s+(?:[\w-]+\s+){0,2}?"
_LOOK_OBJ = r"(?:a|another)\s+(?:quick\s+|fresh\s+|closer\s+)?"
_UP_OBJ = r"(?:(?:it|that|this|them)\s+)?"
_CHECK_NOT = (r"(?!\s+(?:in|out)\b)"
              r"(?!\s+(?:(?:my|the|that|this|your|our)\s+)?(?:math|maths|"
              r"arithmetic|calculations?|conversion|sums|working|reasoning|"
              r"spelling)\b)")

_FAMILIES: tuple[tuple[str, str, str, str, frozenset[str]], ...] = (
    ("restart",
     r"restarting|rebooting",
     r"restarted|rebooted",
     r"restart|reboot",
     frozenset({"restart", "reboot", "reset", "upgrade"})),
    ("open",
     r"opening(?!\s+(?:remarks?|statements?|lines?|acts?|night|ceremony|"
     r"move|question|salvo|hours?)\b)|launching|starting\s+(?:\w+\s+)?up\b",
     r"opened|launched|started\s+(?:\w+\s+)?up\b",
     r"open(?!\s+up\s+about\b)|launch|start\s+(?:\w+\s+)?up\b",
     frozenset({"open", "launch", "start", "app", "url", "youtube", "netflix",
                "prime", "browser"})),
    ("play",
     r"playing(?!\s+(?:devil'?s\s+advocate|it\s+safe|along|catch[-\s]?up|"
     r"with\s+fire|games\s+with)\b)|queu(?:e)?ing|pausing|resuming|"
     r"skipping(?!\s+(?:breakfast|meals?|lunch|dinner|school|class|ahead)\b)",
     r"played(?!\s+(?:devil'?s\s+advocate|it\s+safe|along)\b)|queued|paused|"
     r"resumed|skipped",
     r"play(?!\s+(?:devil'?s\s+advocate|it\s+safe|along|catch[-\s]?up|"
     r"with\s+fire)\b)|queue|pause|resume|skip",
     frozenset({"play", "music", "song", "track", "queue", "pause", "resume",
                "next", "previous", "skip", "spotify", "youtube", "netflix",
                "video", "media", "apple", "itunes"})),
    ("close",
     r"closing(?!\s+(?:thoughts?|remarks?|notes?|words?|arguments?|"
     r"statements?|time|in)\b)|shutting\s+(?:\w+\s+)?down|killing(?!\s+time\b)",
     r"closed|shut\s+(?:\w+\s+)?down|killed",
     r"close(?!\s+(?:by|with)\b)|shut\s+(?:\w+\s+)?down|kill(?!\s+time\b)",
     frozenset({"close", "kill", "shutdown", "shut", "quit", "exit", "window",
                "stop", "end"})),
    ("switch",
     r"switching(?!\s+(?:gears|topics?|subjects?|tack|sides)\b)",
     r"switched(?!\s+(?:gears|topics?|subjects?|tack|sides)\b)",
     r"switch(?!\s+(?:gears|topics?|subjects?|tack|sides)\b)",
     frozenset({"switch", "mode", "use", "monitor", "window", "headset",
                "speakers", "audio", "focus", "desktop", "source", "profile"})),
    ("move",
     r"moving(?!\s+(?:on|forward|along|swiftly|ahead)\b)|minimi[sz]ing|"
     r"maximi[sz]ing",
     r"moved(?!\s+(?:on|forward|along|ahead)\b)|minimi[sz]ed|maximi[sz]ed",
     r"move(?!\s+(?:on|forward|along|ahead)\b)|minimi[sz]e|maximi[sz]e",
     frozenset({"move", "window", "monitor", "minimize", "minimise",
                "maximize", "maximise", "focus", "snap", "desktop"})),
    ("input",
     r"typing|clicking|pressing",
     r"typed|clicked|pressed",
     r"type|click|press",
     frozenset({"type", "click", "press", "key", "keys", "hotkey", "write",
                "input"})),
    ("search",
     r"searching|looking\s+" + _UP_OBJ + r"up\b",
     r"searched|looked\s+" + _UP_OBJ + r"up\b|googled|"
     r"(?:done|ran|run|did)\s+(?:a|an)\s+(?:quick\s+|brief\s+)?(?:web\s+)?"
     r"search\b",
     r"search|look\s+" + _UP_OBJ + r"up\b|google",
     frozenset({"search", "web", "google", "lookup", "look", "browser",
                "browse", "find", "wiki", "wikipedia", "rag"})),
    # "I've checked the logs" / "Checking now, sir." / "Let me check." What
    # can be checked is open-ended (logs, calendar, weather, the printer), so
    # ANY action that succeeded this turn grounds it (the "*" token).
    # Re-checking its OWN arithmetic is not an action ("Let me double-check
    # my math: 100 °F is 37.8 °C"), and "checking account" is a noun.
    ("check",
     r"(?:double[-\s]?)?checking" + _CHECK_NOT + r"(?!\s+accounts?\b)|"
     r"verifying" + _CHECK_NOT,
     r"(?:double[-\s]?)?checked" + _CHECK_NOT + r"|verified" + _CHECK_NOT,
     r"(?:double[-\s]?)?check" + _CHECK_NOT + r"|verify" + _CHECK_NOT,
     frozenset({"*"})),
    ("timer",
     r"setting\s+(?:up\s+)?" + _TIMER_OBJ + r"(?:timer|alarm|reminder)\b|"
     r"starting\s+" + _TIMER_OBJ + r"timer\b",
     r"set\s+(?:up\s+)?" + _TIMER_OBJ + r"(?:timer|alarm|reminder)\b|"
     r"started\s+" + _TIMER_OBJ + r"timer\b",
     r"set\s+(?:up\s+)?" + _TIMER_OBJ + r"(?:timer|alarm|reminder)\b|"
     r"start\s+" + _TIMER_OBJ + r"timer\b|remind\s+you\b",
     frozenset({"timer", "alarm", "reminder", "remind", "schedule"})),
    ("login",
     r"logging\s+(?:you\s+)?(?:in|out|off)\b",
     r"logged\s+(?:you\s+)?(?:in|out|off)\b",
     r"log\s+(?:you\s+)?(?:in|out|off)\b",
     frozenset({"login", "logout", "log", "sign", "signin"})),
    ("look",
     r"taking\s+" + _LOOK_OBJ + r"(?:screen\s?shot|look|peek|glance)\b",
     r"(?:taken|took)\s+" + _LOOK_OBJ + r"(?:screen\s?shot|look|peek|glance)\b",
     r"take\s+" + _LOOK_OBJ + r"(?:screen\s?shot|look|peek|glance)\b",
     frozenset({"screenshot", "see", "screen", "look", "vision", "describe",
                "camera", "kinect", "glance", "capture"})),
    ("turn",
     r"turning\s+" + _TURN_OBJ + r"(?:on|off|up|down)\b",
     r"turned\s+" + _TURN_OBJ + r"(?:on|off|up|down)\b",
     r"turn\s+" + _TURN_OBJ + r"(?:on|off|up|down)\b",
     frozenset({"turn", "light", "lights", "lamp", "smart", "home", "device",
                "control", "volume", "plug", "power", "fan", "tv", "mode",
                "hue", "kasa", "govee", "lifx", "tuya"})),
    ("send",
     r"sending",
     r"sent",
     r"send",
     frozenset({"send", "email", "message", "text", "reply", "sms", "post",
                "draft", "mail"})),
    ("mute",
     r"muting|unmuting",
     r"muted|unmuted",
     r"mute|unmute",
     frozenset({"mute", "unmute", "volume", "sound", "audio", "mic"})),
)

_ADVERBS = (r"(?:(?:now|just|currently|already|also|quickly|immediately|"
            r"successfully)\s+)?")


def _compile_family(gerund: str, participle: str, base: str):
    progressive = re.compile(
        r"\b(?:i'?m|i\s+am)\s+" + _ADVERBS + r"(?:" + gerund + r")\b")
    # "I've taken the liberty of searching ..." is the persona's initiative
    # opener wrapped round a gerund: as much a claim as "I've searched".
    perfect = re.compile(
        r"\b(?:i'?ve|i\s+have|i)\s+"
        r"(?:(?:just|now|already|also|successfully|gone\s+ahead\s+and|"
        r"went\s+ahead\s+and)\s+)?(?:" + participle + r")\b"
        r"|\b(?:i'?ve|i\s+have|i'?m|i\s+am|i)\s+(?:just\s+|already\s+)?"
        r"(?:taken|took|taking)\s+the\s+liberty\s+of\s+" + _ADVERBS +
        r"(?:" + gerund + r")\b")
    future = re.compile(
        r"\b(?:i'?ll|i\s+will|i'?m\s+going\s+to|i\s+am\s+going\s+to|"
        r"i'?m\s+about\s+to|let\s+me)\s+"
        r"(?:(?:now|just|quickly|also|immediately|go\s+ahead\s+and)\s+)?"
        r"(?:" + base + r")\b")
    # Subjectless narration: a clause that STARTS with the gerund ("Moving it
    # now, sir.") or the participle ("Sent, sir.", "Queued three songs.").
    # A participle followed by a preposition is a passive / participial
    # opener ("Opened in 1889, ...", "Closed on Sundays"), not a claim.
    narration = re.compile(
        r"(?:" + gerund + r")\b"
        r"|(?:" + participle + r")\b(?!\s+(?:by|in|on|at|since|for|during|"
        r"from|between|over|under|with|to)\b)")
    return progressive, perfect, future, narration


_COMPILED = tuple(
    (name, _compile_family(g, p, b), tokens)
    for name, g, p, b, tokens in _FAMILIES
)

# ── clause structure ─────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"\[[^\]]*\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
# Clause breaks inside a sentence: ; — – " - " "..." and a colon that is not
# part of a clock time ("2:53").
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:;|(?<!\d):(?!\d)|[—–]|\s-\s|\.{3})\s*")

# Courtesy lead-ins peeled off the front of a clause before looking for a
# clause-initial gerund or an acknowledgement ("Very good, sir, opening ...").
# Pure courtesy — never claims by themselves. "right" never eats "right away".
_LEADIN_RE = re.compile(
    r"^(?:very\s+good|very\s+well|certainly|of\s+course|as\s+you\s+wish|"
    r"indeed|understood|quite\s+right|quite\s+so|right(?!\s+away)|alright|"
    r"all\s+right|okay|ok|sure(?:\s+thing)?|absolutely|yes|yep|no\s+problem|"
    r"with\s+pleasure|excellent|splendid|perfect|noted|good|great|ah|oh|"
    r"well(?!\s+(?:played|done)\b)|so|and|now|then|next|first|also|sir|"
    r"if\s+i\s+may(?:\s+say\s+so)?)"
    r"(?:\s*,?\s*sir\b)?(?:\s*[,!.…]+\s*|\s+|$)")

# Execution acknowledgements. "done"-type acks report completion (grounded by
# any action that succeeded this turn); the rest promise action (never
# grounded when bare).
_DONE_ACKS = r"done|all\s+done|consider\s+it\s+done|(?:it'?s|that'?s)\s+done"
_PENDING_ACKS = (r"on\s+it|right\s+away|straight\s+away|at\s+once|will\s+do|"
                 r"working\s+on\s+it|coming\s+(?:right\s+)?up")
_ACK_RE = re.compile(
    r"^(?:(?P<done>" + _DONE_ACKS + r")|(?P<pending>" + _PENDING_ACKS + r"))"
    r"(?:(?P<sir>\s*,?\s*sir\b)\s*(?:[,.!…]+\s*|$)|\s*(?:[.!…]+\s*|$))")

# A spoken ack PREFACE on the original-case reply (for strip_ack_preface).
_ACK_PREFACE_RE = re.compile(
    r"^\s*(?:" + _DONE_ACKS + r"|" + _PENDING_ACKS + r")"
    r"(?:\s*,?\s*sir\b\s*(?:[.!,…—–;:]+|\s-\s)|"
    r"\s*(?:[.!…—–;:]+|\s-\s))\s*",
    re.IGNORECASE)

# Offers / conditionals / questions back to the owner are never claims.
_OFFER_RE = re.compile(
    r"\b(?:shall\s+i|should\s+i|want\s+me\s+to|would\s+you\s+like|"
    r"if\s+you(?:'d|\s+would)?\s+(?:like|wish|want|prefer)|"
    r"should\s+you\s+(?:wish|want|like)|if\s+needed|if\s+necessary|"
    r"whenever\s+you|once\s+you|when\s+you(?:'re|\s+are)?\s+ready|"
    r"say\s+the\s+word)\b")

# Promise of a future / in-progress action inside otherwise-plain content.
_PROMISE_RE = re.compile(
    r"\b(?:i'?ll|i\s+will|i'?m\s+going\s+to|let\s+me(?!\s+know\b)|"
    r"give\s+me\s+(?:a|one)\s+(?:moment|second|sec|minute|tick)|"
    r"one\s+moment|just\s+a\s+(?:moment|second|sec|tick)|stand\s+by|"
    r"bear\s+with\s+me|shortly|momentarily|"
    r"in\s+a\s+(?:moment|second|sec|minute|jiffy|tick)|"
    r"any\s+(?:moment|second)\s+now|on\s+(?:its|the)\s+way|under\s*way|"
    r"in\s+progress|coming\s+(?:right\s+)?up|"
    r"(?:should|will|it'?ll)\s+be\s+(?:up|ready|done|open|on|off|playing|"
    r"running|back|finished))\b")

# Narration signals that make a clause-initial gerund JARVIS's own narration
# even when a copula follows ("Playing 'Can't Stop' now, sir").
_NARRATION_SIGNAL_RE = re.compile(
    r"\b(?:now|for\s+you|right\s+away|at\s+once|straight\s+away|shortly|"
    r"momentarily|as\s+requested|as\s+we\s+speak|"
    r"in\s+a\s+(?:moment|second|sec|jiffy|tick))\b")

# A finite verb after a clause-initial gerund phrase marks the gerund as a
# noun SUBJECT ("Moving house IS stressful", "Playing chess CAN help").
# A gerund subject is singular, so a lexical verb must carry the -s ("takes",
# not "take" — "Playing Take Five" is narration, "Playing chess takes years"
# is a fact).
_SUBJECT_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|isn'?t|aren'?t|wasn'?t|weren'?t|can|could|may|"
    r"might|must|should|would|will|has|had|does|did|doesn'?t|"
    r"tends|helps|makes|takes|requires|means|remains|becomes|"
    r"seems|costs|burns|improves|reduces|increases|boosts|lowers|"
    r"raises|causes|keeps|gives|counts|matters|releases|produces|"
    r"creates|lets|allows|prevents|affects|triggers|involves)\b")
_CLAUSE_CUT_RE = re.compile(
    r"\b(?:that|which|who|whom|whose|where|when|while|because|so|and|but|or|"
    r"then|until|before|after|if|as)\b")
_QUOTED_RE = re.compile(r"(?:(?<=\s)|^)[\"'“‘][^\"'”’]{1,80}"
                        r"[\"'”’](?=\W|$)")
# A new third-party subject after the first comma: "Moving about 4 cm a year,
# THE MOON drifts ..." — a participial opener, not narration.
_NEW_SUBJECT_RE = re.compile(
    r"^(?:the|a|an|this|these|those|its|their|his|her|it|they|he|she|we|"
    r"you|there|people|scientists|researchers|one|most|many|some)\b")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    t = _TAG_RE.sub(" ", text or "")
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("*", " ").replace("_", " ")
    return re.sub(r"[ \t]+", " ", t).strip().lower()


def _strip_leadins(core: str) -> str:
    prev = None
    while prev != core:
        prev = core
        m = _LEADIN_RE.match(core)
        if m and m.end() > 0:
            core = core[m.end():].lstrip(" ,")
    return core


def _substantive(core: str) -> bool:
    words = [w for w in _WORD_RE.findall(core) if w != "sir"]
    return len(words) >= 2


def _gerund_is_noun_use(core: str, end: int) -> bool:
    """True when the clause-initial gerund ending at ``end`` is a noun subject
    or a participial opener rather than JARVIS narrating its own action."""
    rest = core[end:]
    comma = rest.find(",")
    head = rest if comma < 0 else rest[:comma]
    tail = "" if comma < 0 else rest[comma + 1:]
    if _NARRATION_SIGNAL_RE.search(head):
        return False
    head_cut = _CLAUSE_CUT_RE.split(_QUOTED_RE.sub(" ", head), maxsplit=1)[0]
    if _SUBJECT_VERB_RE.search(head_cut):
        return True
    if head.strip() and tail:
        t = _strip_leadins(tail.strip(" ,"))
        if _NEW_SUBJECT_RE.match(t) and not _PROMISE_RE.search(t):
            return True
    return False


# First-person idioms that name a verb without claiming the act ("Last time I
# checked, Pluto was a dwarf planet"). Blanked before the family scan.
_IDIOM_RE = re.compile(
    r"\b(?:(?:the\s+)?last\s+time\s+i\s+checked|when\s+i\s+last\s+checked)\b")


def _clause_claim(core: str) -> Optional[tuple[str, str]]:
    """(family, matched text) for a first-person action claim in one clause
    (lead-ins already stripped), else None."""
    if core.endswith("?") or _OFFER_RE.search(core):
        return None
    core = _IDIOM_RE.sub(" ", core).strip(" ,")
    if not core:
        return None
    # Narration may also open a later comma-separated part ("Excellent
    # choice, sir, playing it now."), but only with an explicit narration
    # signal there — "The moon, moving at 4 cm a year, ..." has none.
    later = [p for p in (_strip_leadins(x.strip()) for x in core.split(",")[1:])
             if p and _NARRATION_SIGNAL_RE.search(p)]
    for name, (progressive, perfect, future, narration), _tokens in _COMPILED:
        for rx in (progressive, perfect, future):
            m = rx.search(core)
            if m:
                return name, m.group(0)
        m = narration.match(core)
        if m and not _gerund_is_noun_use(core, m.end()):
            return name, m.group(0)
        for part in later:
            m = narration.match(part)
            if m and not _gerund_is_noun_use(part, m.end()):
                return name, m.group(0)
    return None


def _ran_tokens(ran_actions: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for n in ran_actions or ():
        low = str(n).lower()
        tokens.add(low)
        tokens.update(t for t in re.split(r"[_\W]+", low) if t)
    return tokens


def _family_tokens(name: str) -> frozenset[str]:
    for fam, _rx, tokens in _COMPILED:
        if fam == name:
            return tokens
    return frozenset()


def _segments(text: str):
    """Yield (ack_kind, ack_text, core) for each clause of ``text``: ack_kind
    is 'done' / 'pending' when the clause opened with an execution ack (else
    None), ack_text the ack as written, and core what remains after lead-ins
    and the ack."""
    for sentence in _SENTENCE_SPLIT_RE.split(_norm(text)):
        if not sentence.strip():
            continue
        is_question = sentence.rstrip().endswith("?")
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            core = _strip_leadins(clause.strip())
            ack = ack_text = None
            m = _ACK_RE.match(core)
            if m:
                ack = "done" if m.group("done") else "pending"
                ack_text = re.sub(r"\s+", " ", m.group(ack))
                core = _strip_leadins(core[m.end():].strip(" ,"))
            if is_question and core and not core.endswith("?"):
                core = core.rstrip(".!") + "?"
            yield ack, ack_text, core.strip()


# ── owner question detection ─────────────────────────────────────────────────
_Q_FILLER_RE = re.compile(
    r"^(?:(?:hey|so|and|ok(?:ay)?|um+|uh+|well|but|also|then|jarvis|sir|"
    r"quick\s+question|one\s+more\s+thing)[,\s]+)+")
_PRON = (r"(?:i|you|we|they|he|she|it|there|that|this|anyone|anybody|"
         r"anything|everything|someone|something)")
_DET = r"(?:the|my|your|our|his|her|its|their|a|an|any)"
_QUESTION_START_RE = re.compile(
    r"^(?:what|what'?s|whats|who|who'?s|whose|whom|when|when'?s|where|"
    r"where'?s|why|which|how|how'?s|how'?d)\b"
    r"|^(?:is|isn'?t|are|aren'?t|was|wasn'?t|were|weren'?t|am|does|doesn'?t|"
    r"did|didn'?t|has|hasn'?t|had|should|shall|may|might)\s+"
    r"(?:" + _PRON + r"|" + _DET + r")\b"
    # "do"/"have" + a noun phrase is an imperative ("do the dishes", "have a
    # look"); only "do you ..." / "have you ..." is a question.
    r"|^(?:do|don'?t|have|haven'?t)\s+(?:i|you|we|they|he|she)\b"
    r"|^(?:can|could|would|will)\s+(?:i|we|it|there|they|he|she|" + _DET + r")\b"
    r"|^(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:tell\s+me|explain|"
    r"describe|remind\s+me\s+(?:what|who|when|where|why|how|which))\b"
    r"|^tell\s+me(?!\s+(?:when|if|once|as\s+soon))\b"
    r"|^remind\s+me\s+(?:what|who|when|where|why|how|which)\b"
    r"|^(?:explain|describe|define)\b"
    r"|^(?:do\s+you\s+(?:know|remember)|any\s+idea)\b")
_POLITE_COMMAND_RE = re.compile(
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?(?!tell\b|explain\b|"
    r"describe\b|remind\s+me\s+(?:what|who|when|where|why|how|which)\b)")


def looks_like_question(user_text: str) -> bool:
    """True when the owner's utterance is an information request (a question
    or "tell me ..."), not a command. Voice transcripts often lack '?', so the
    opening words decide; "can you open X" is a polite COMMAND."""
    t = _norm(user_text)
    if not t:
        return False
    t = _Q_FILLER_RE.sub("", t).strip()
    if not t:
        return False
    if _POLITE_COMMAND_RE.match(t):
        return False
    return bool(_QUESTION_START_RE.match(t) or t.endswith("?"))


# ── the detector ─────────────────────────────────────────────────────────────
def find_unverified_claim(text: str, *, ran_actions: Iterable[str] = (),
                          user_text: str = "") -> Optional[str]:
    """Return the claim phrase when ``text`` (a reply that carried NO
    ``[ACTION:]`` token) claims JARVIS acted and nothing this turn grounds it;
    None when the reply is not an ungrounded execution claim.

    ``ran_actions`` — names of the actions that already ran SUCCESSFULLY this
    turn (empty on the first round). ``user_text`` — the owner's utterance for
    this turn ("" when unknown: an acknowledgement is then treated as a claim,
    exactly as before).
    """
    if not text or not text.strip():
        return None
    ran = _ran_tokens(ran_actions)
    any_ran = bool(ran)
    acks: list[tuple[str, str]] = []      # (kind, phrase)
    content: list[str] = []               # substantive non-claim cores
    for ack, ack_text, core in _segments(text):
        if ack:
            acks.append((ack, ack_text))
        if not core:
            continue
        claim = _clause_claim(core)
        if claim:
            fam, phrase = claim
            tokens = _family_tokens(fam)
            if not (tokens & ran or ("*" in tokens and any_ran)):
                return phrase.strip()
            content.append(core)     # a grounded summary of a real result
            continue
        if _substantive(core):
            content.append(core)
    if not acks:
        return None
    kind, phrase = acks[0]
    promise = any(_PROMISE_RE.search(c) for c in content)
    if not content:
        # Bare acknowledgement: only a completion ack after a real action.
        if any_ran and all(k == "done" for k, _ in acks):
            return None
        return phrase
    if promise:
        return phrase
    if any_ran or looks_like_question(user_text):
        # A summary of this turn's results, or a filler preface on an answer.
        return None
    return phrase


# "I'm running the numbers now, sir" / "I am still reading the page": JARVIS
# narrating its own LOOK-UP work in progress. A short list of work verbs on
# purpose - "I'm playing it now, sir" reports a result, it is not a promise.
_WORK_PROGRESS_RE = re.compile(
    r"^(?:i'?m|i\s+am)\s+(?:now\s+|just\s+|currently\s+|still\s+|already\s+)?"
    r"(?:running|working|checking|looking|reading|searching|scanning|"
    r"calculating|computing|crunching|cross[-\s]?referencing|analy[sz]ing|"
    r"reviewing|processing|gathering|pulling|fetching|digging|trying|"
    r"attempting|thinking|compiling|going\s+through)\b")
# A promise glued onto content ("the printer is offline, I'll keep trying")
# is split off so the content still counts.
_PROMISE_SPLIT_RE = re.compile(
    r"(?:,\s*|\s+and\s+|\s+but\s+)"
    r"(?=(?:i'?ll|i\s+will|let\s+me|i'?m\s+going\s+to)\b)")
_DIGIT_RE = re.compile(r"\d")


def is_progress_only(text: str) -> bool:
    """True when ``text`` tells the owner nothing beyond "I'm on it": it is
    empty, or every clause is a pending acknowledgement ("On it, sir"), a
    promise of a further step ("I'll have those results for you in a
    moment", "Let me take a look", "One moment") or narration of look-up work
    in progress ("I'm running the numbers now"). False as soon as one clause
    carries anything else - an answer, a figure, a report ("Done, sir",
    "Playing it now, sir"), a refusal ("I'm afraid the printer isn't
    reachable") or a question back to the owner. Used to decide whether a
    follow-up chain that stopped early owes the owner a close-out line."""
    if not text or not text.strip():
        return True
    for ack, _ack_text, core in _segments(text):
        if ack == "done":
            return False
        if not core:
            continue
        if core.endswith("?"):
            return False
        for part in _PROMISE_SPLIT_RE.split(core):
            part = _strip_leadins(part.strip(" ,"))
            if not part:
                continue
            if _DIGIT_RE.search(part):
                return False
            if _WORK_PROGRESS_RE.match(part) or _PROMISE_RE.search(part):
                continue
            if _substantive(part):
                return False
    return True


def strip_ack_preface(text: str, user_text: str, *,
                      ran_actions: Iterable[str] = ()) -> str:
    """Drop a meaningless leading execution ack ("On it, sir.") from a reply
    that ANSWERS the owner's question. Leading bracket tags ([intent:...]) are
    kept. Returns ``text`` unchanged unless the owner asked a question, the
    reply opens with an ack, and the rest is a substantive answer with no
    ungrounded claim (``ran_actions`` as for find_unverified_claim) and no
    promise."""
    if not text or not looks_like_question(user_text):
        return text
    m_tags = re.match(r"^(?:\s*\[[^\]]*\]\s*)+", text)
    tags = m_tags.group(0) if m_tags else ""
    prose = text[len(tags):]
    m = _ACK_PREFACE_RE.match(prose)
    if not m:
        return text
    rest = prose[m.end():].lstrip()
    if not rest or not _substantive(_norm(rest)):
        return text
    if find_unverified_claim(rest, ran_actions=ran_actions,
                             user_text=user_text) is not None:
        return text
    if _PROMISE_RE.search(_norm(rest)):
        return text
    rest = rest[0].upper() + rest[1:]
    return f"{tags.strip()} {rest}".strip() if tags.strip() else rest
