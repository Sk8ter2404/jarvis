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
  4. A PASSIVE completion claim about the owner's target (2026-10-02 live:
     "Very good, sir. <app> has been closed." with nothing run): "<subject>
     has / have / 's been <participle>", "is now <participle>", "was
     successfully <participle>", when the subject is a pronoun or quantifier,
     a word of the owner's utterance, or a thing JARVIS acts on. Not when the
     owner's turn asks about state ("did you close it?", "has it been sent",
     "check whether it's closed"), thanks JARVIS or reports news ("thanks for
     closing that", "the meeting got moved" - _owner_states), the clause is
     dated ("since this morning", "an hour ago", "for decades") or has a
     passive agent ("closed by the court"). See _passive_claim.

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
sir.") is never grounded — it promises something new. The OWNER's words
never ground a claim: Parakeet writes "close" as "closed", and "Jarvis closed
notepad" is a misheard command, not an action that ran.

Pure: no I/O, no monolith import, stdlib ``re`` only, so it is testable on the
light-deps CI runner (tests/test_claim_validator.py).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

__all__ = [
    "asks_owner",
    "find_completed_claim",
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
    # "I've run the calculations, sir." / "Cross-referencing now." / "I've
    # crunched the numbers" (2026-10-01): stock completed-work lines spoken
    # live with no action run. Any action that ran this turn grounds them
    # ("*"); JARVIS's own arithmetic with its result in the reply is not a
    # claim (_ARITH_RESULT_RE in find_unverified_claim).
    ("calculate",
     r"calculating|computing|crunching\s+(?:the\s+|some\s+)?numbers|"
     r"running\s+(?:the\s+|some\s+)?(?:numbers|calculations?|figures|maths?)|"
     r"cross[-\s]?referencing|number[-\s]crunching",
     r"calculated|computed|crunched\s+(?:the\s+|some\s+)?numbers|"
     r"(?:ran|run)\s+(?:the\s+|some\s+)?(?:numbers|calculations?|figures|"
     r"maths?)|cross[-\s]?referenced|"
     r"done\s+the\s+(?:maths?|calculations?|numbers|sums)",
     r"calculate|compute|crunch\s+(?:the\s+|some\s+)?numbers|"
     r"run\s+(?:the\s+|some\s+)?(?:numbers|calculations?|figures|maths?)|"
     r"cross[-\s]?reference",
     frozenset({"*"})),
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

# JARVIS stating the result of its OWN arithmetic: "fifteen percent of eighty
# is twelve", "I've calculated it, sir: 391". A calculate-family claim in a
# reply that carries one is the working shown, not an invented action.
#
# Review repair (2026-10-02): "one" is not a figure ("it's one of the trickier
# trade-offs" read as a result and let the live shape through), and neither is
# any number word followed by "of" ("two of them").
_NUMBER_WORDS = (r"(?:zero|two|three|four|five|six|seven|eight|nine|ten|"
                 r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
                 r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                 r"eighty|ninety|hundred|thousand|million|billion)")
_ARITH_RESULT_RE = re.compile(
    r"(?:\bis|'s|\bequals|=|\bcomes\s+(?:out\s+)?(?:to|at)|"
    r"\bworks\s+out\s+(?:to|at)|\bgives|\bmakes|:)\s*"
    r"(?:about\s+|roughly\s+|approximately\s+|exactly\s+|around\s+)?"
    r"(?:-?\d|(?:" + _NUMBER_WORDS + r"|a\s+half)\b(?!\s+of\b))")
# ...or an amount with its unit anywhere in the reply ("you'd need about forty
# dollars a week", "roughly 3.5 hours"). A bare model size ("the 32B") is not.
_AMOUNT_RE = re.compile(
    r"(?:\$\s*\d|\b(?:\d[\d,.]*|" + _NUMBER_WORDS + r"|a\s+half)\s*"
    r"(?:%|percent\b|dollars?\b|bucks\b|cents?\b|pounds?\b|euros?\b|"
    r"hours?\b|minutes?\b|mins?\b|seconds?\b|days?\b|weeks?\b|months?\b|"
    r"years?\b|pages?\b|gigabytes?\b|gigs?\b|megabytes?\b|terabytes?\b|"
    r"gb\b|mb\b|tb\b|kilo\w*|miles?\b|feet\b|foot\b|inches?\b|"
    r"met(?:er|re)s?\b|degrees?\b|watts?\b|kwh\b|tokens?\b|times\b))")
# The owner's question had numbers in it, so there was something to work out:
# "I've run the numbers" over a quantitative ask is mental arithmetic, not an
# invented action ("if I read 30 pages a day, when do I finish?").
_QUANT_ASK_RE = re.compile(
    r"\d|\b(?:" + _NUMBER_WORDS + r"|half|quarter|percent|percentage|twice|"
    r"double|triple|dozen)\b")
# A calculate-family gerund / participle that IS the whole clause ("Calculating,
# sir.", "Running the numbers now.") - the phrasebook's working line. Followed
# by anything else it is a noun or an adjective ("Computing power", "Calculated
# risk", "Cross-referencing tools"), never narration.
_BARE_REST_RE = re.compile(
    r"^(?:[\s,.!…]|now\b|sir\b|for\s+you\b|as\s+we\s+speak\b|"
    r"right\s+away\b)*$")


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
        if name == "calculate":
            got = _calculate_claim(core, progressive, perfect, future,
                                   narration)
            if got:
                return got
            continue
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


# ── passive completion claims (2026-10-02 live) ──────────────────────────────
# Parakeet writes the imperative "close" as "closed", so "Jarvis, close <app>"
# reached the brain as a past-tense sentence, and the local model answered in
# the PASSIVE voice - "Very good, sir. <app> has been closed." - with no action
# token. No first-person rule above sees that shape, so nothing ran and nothing
# was corrected. Rule: "<subject> has / have / 's been <participle>" ("is now
# <participle>", "was successfully <participle>") is a claim when its subject
# is the owner's target: a pronoun or quantifier ("it", "that", "everything",
# "all the windows", "your email"), a word of the owner's own utterance, or a
# thing JARVIS acts on (a window, an app, a timer, a message, the lights ...).
# Never a claim: a question, an owner turn that is itself a question (a status
# or recall question - "did you close it?" - answered from the conversation),
# a clause dated in the past ("since this morning", "an hour ago", "for
# decades"), a passive agent ("closed by the court"), or a report of an action
# that DID run this turn (grounded by family, exactly as the active forms).
# The owner's words never ground anything: "Jarvis closed notepad" is what he
# SAID, not an action that ran.
_PASSIVE_FAMILIES: tuple[tuple[str, str], ...] = (
    ("close", r"closed|shut\s+down|killed|terminated|quit|stopped|ended"),
    ("open", r"opened|launched|started(?:\s+up)?"),
    ("play", r"played|queued|paused|resumed|skipped"),
    ("switch", r"switched(?:\s+(?:on|off|over))?"),
    ("move", r"moved|minimi[sz]ed|maximi[sz]ed|snapped"),
    ("send", r"sent|emailed|messaged|forwarded|delivered|posted"),
    ("mute", r"muted|unmuted|silenced"),
    ("restart", r"restarted|rebooted"),
    ("turn", r"turned\s+(?:on|off|up|down)"),
    ("timer", r"set(?:\s+up)?|scheduled|created|added"),
    # Generic completion: any action that ran this turn grounds it.
    ("done", r"done|completed|finished|handled|sorted(?:\s+out)?|"
             r"taken\s+care\s+of|carried\s+out|executed|actioned"),
)
_PASSIVE_TOKENS: dict[str, frozenset[str]] = {
    "timer": frozenset({"timer", "alarm", "reminder", "remind", "schedule",
                        "set", "volume", "brightness", "add", "create"}),
    "done": frozenset({"*"}),
}
_PASSIVE_ADV = r"(?:(?:now|just|already|all|also|successfully|properly)\s+)*"
_PASSIVE_RES = tuple(
    (fam, re.compile(
        r"(?P<aux>(?:\s+(?:has|have)|'s|'ve)\s+" + _PASSIVE_ADV + r"been\s+"
        r"|(?:\s+(?:is|are)|'s|'re)\s+now\s+"
        r"|\s+(?:was|were)\s+(?:successfully|just)\s+)" + _PASSIVE_ADV +
        r"(?P<part>" + parts + r")\b"))
    for fam, parts in _PASSIVE_FAMILIES)
# What follows the participle when the clause is not JARVIS's own work: a
# passive agent ("closed by the court") or a past date / span ("since 2019",
# "an hour ago", "for decades", "earlier today").
_PASSIVE_NOT_NOW_RE = re.compile(
    r"^\s*by\b|\b(?:since|ago|earlier|yesterday|previously|before\s+you|"
    r"last\s+(?:night|week|month|year|time)|this\s+(?:morning|afternoon|week)|"
    r"years?|decades?|centur(?:y|ies)|months|ages|in\s+\d{4})\b")
# Subjects that ARE the owner's target whatever the turn said.
_PASSIVE_PRONOUN_RE = re.compile(
    r"^(?:it|that|this|they|those|these|everything|everyone|all|both|each|"
    r"every|your|the\s+rest|the\s+others?|the\s+remaining)\b")
_PASSIVE_TARGET_RE = re.compile(
    r"\b(?:windows?|apps?|applications?|programs?|tabs?|browsers?|timers?|"
    r"alarms?|reminders?|e-?mails?|messages?|texts?|lights?|lamps?|music|"
    r"songs?|tracks?|playlists?|videos?|volume|screenshots?|files?|"
    r"folders?|requests?|tasks?|commands?|orders?|notes?)\b")
_PASSIVE_SUBJ_CUT_RE = re.compile(r",|\b(?:and|so|then|but)\b")
_PASSIVE_STOP = frozenset({
    "the", "a", "an", "my", "your", "our", "this", "that", "these", "those",
    "all", "and", "for", "jarvis", "sir", "please", "now", "just", "with",
    "from", "into", "onto", "has", "have", "been", "was", "were", "are",
    "except", "else", "then", "too", "also", "can", "you", "it", "its",
})


# The owner asking about state ("has notepad been closed", "is it sent") even
# without a "?" - looks_like_question needs a pronoun or determiner after the
# auxiliary. "had been at close" (a misheard command) is not a question.
_OWNER_STATUS_ASK_RE = re.compile(
    r"^(?:has|have|had|is|are|was|were)\s+(?!been\b|being\b)\S+")


# "check whether notepad has been closed" asks about state too (review
# 2026-10-02): the follow-up's "Notepad has been closed, sir" after
# list_windows reports what it found.
_OWNER_CHECK_ASK_RE = re.compile(
    r"^(?:(?:double[-\s]?)?check|see|find\s+out|confirm|verify)\s+"
    r"(?:whether|if)\b")


def _owner_asks_status(user_text: str) -> bool:
    if looks_like_question(user_text):
        return True
    t = _Q_FILLER_RE.sub("", _norm(user_text)).strip()
    return bool(_OWNER_STATUS_ASK_RE.match(t) or _OWNER_CHECK_ASK_RE.match(t))


# The owner THANKING JARVIS or TELLING it news, not asking or commanding
# (review 2026-10-02): "thanks for closing that", "the meeting got moved", "I
# heard the shop shut down". JARVIS echoing it in the passive ("It has been
# closed, sir", "it has been moved to Thursday") claims nothing it did - and a
# flagged reply is withheld and re-prompted ("emit the real action"), so
# reading it as a claim silences a true reply and invites the "can't actually
# move the moon" retraction, or a second close of a window the owner reopened.
#   * thanks / praise opens a statement;
#   * otherwise a subject or determiner opener ("I", "the", "my", "it") with a
#     PAST verb after it ("the meeting GOT moved", "my flight LANDED", "I
#     HEARD ..."). A present-tense remark stays a possible request ("it's too
#     dark in here", "I'm bored", "I'm done with notepad" - an -ed word after
#     a copula is an adjective) and so does an elliptical one ("the other one
#     too"); commands open with their verb ("close it", Parakeet's "Jarvis
#     closed notepad") or a request lead ("I need you to ...").
_OWNER_THANKS_RE = re.compile(
    r"^(?:thanks|thank|cheers|ta|good|great|nice|lovely|perfect|excellent|"
    r"brilliant|awesome|cool|wow|well\s+done|nicely\s+done)\b")
_OWNER_OPENER_RE = re.compile(
    r"^(?:i(?!'m\b|m\b)|i've|ive|we(?!'re\b)|we've|he|she|they(?!'re\b)|"
    r"you(?!'re\b)|the|a|an|my|our|your|his|her|their|it(?!'s\b)|its|"
    r"that(?!'s\b)|this|these|those|there(?!'s\b)|apparently|guess|looks|"
    r"seems)\b")
_OWNER_COMMAND_LEAD_RE = re.compile(
    r"^(?:i|we)\s+(?:need|want|would\s+like|'d\s+like|d\s+like)\b"
    r"|^you\s+(?:can|could|should|may|might|must|need|have\s+to|will|"
    r"would)\b")
_OWNER_PAST_RE = re.compile(
    r"\b(?:got|was|were|had|came|went|did|heard|saw|sent|made|took|left|"
    r"found|told|said|thought|knew|finally|already|[a-z]{3,}ed)\b")


def _owner_states(user_text: str) -> bool:
    """True when the owner's turn thanks JARVIS or reports news (a past
    event) rather than commanding or asking - see _OWNER_THANKS_RE."""
    t = _Q_FILLER_RE.sub("", _norm(user_text)).strip()
    if not t or _OWNER_COMMAND_LEAD_RE.match(t):
        return False
    if _OWNER_THANKS_RE.match(t):
        return True
    return bool(_OWNER_OPENER_RE.match(t) and _OWNER_PAST_RE.search(t))


def _passive_words(text: str) -> set[str]:
    out = set()
    for w in _WORD_RE.findall(text or ""):
        if len(w) >= 3 and w not in _PASSIVE_STOP:
            out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def _passive_claim(core: str, owner_words: set[str]
                   ) -> Optional[tuple[str, str]]:
    """(family, matched text) for a passive completion claim about the
    owner's target in one clause (lead-ins already stripped), else None.
    ``owner_words`` - _passive_words of the owner's utterance."""
    if core.endswith("?") or _OFFER_RE.search(core):
        return None
    for fam, rx in _PASSIVE_RES:
        m = rx.search(core)
        if not m:
            continue
        if _PASSIVE_NOT_NOW_RE.search(core[m.end():]):
            return None
        subj = _PASSIVE_SUBJ_CUT_RE.split(core[:m.start()])[-1].strip()
        if not subj or len(subj.split()) > 8:
            continue
        if (_PASSIVE_PRONOUN_RE.match(subj) or _PASSIVE_TARGET_RE.search(subj)
                or (_passive_words(subj) & owner_words)):
            return fam, (subj + m.group(0)).strip()
    return None


_PARTICIPLE_RES = {name: re.compile(r"(?:" + participle + r")")
                   for name, _g, participle, _b, _t in _FAMILIES}


def _clause_completed_claim(core: str) -> Optional[tuple[str, str]]:
    """(family, matched text) when one clause (lead-ins stripped) claims an
    action ALREADY happened: a perfect / past form ("I've closed it", "I sent
    it", "I've taken the liberty of closing ...") or a clause-initial
    participle ("Sent, sir."). Progressive and future forms ("Opening it
    now", "I'll send it") are not completions. Else None."""
    if core.endswith("?") or _OFFER_RE.search(core):
        return None
    core = _IDIOM_RE.sub(" ", core).strip(" ,")
    if not core:
        return None
    for name, (_progressive, perfect, _future, narration), _t in _COMPILED:
        m = perfect.search(core)
        if m:
            return name, m.group(0)
        m = narration.match(core)
        if (m and _PARTICIPLE_RES[name].fullmatch(m.group(0))
                and not _gerund_is_noun_use(core, m.end())):
            return name, m.group(0)
    return None


def _grounded(tokens, ran: set[str], any_ran: bool, phrase: str) -> bool:
    """A claim of a family with ``tokens`` is grounded by an action of that
    family, by any action for a "*" family, or by an action named after the
    claim's own words ("the print job has been sent" after print_document)."""
    tokens = tokens or frozenset()
    return bool(tokens & ran or ("*" in tokens and any_ran)
                or (_passive_words(phrase) & ran))


def find_completed_claim(text: str, *, ran_actions: Iterable[str] = (),
                         user_text: str = "") -> Optional[str]:
    """The phrase when ``text`` claims an action ALREADY HAPPENED and nothing
    this turn grounds it: a perfect / past first-person claim, a clause-
    initial participle, a passive completion (see _passive_claim) or a bare
    "Done, sir." with nothing run. None otherwise - including progressive
    and future narration ("Opening it now, sir.", "I'll send it"), which is
    how a reply honestly leads into an action token it is about to carry.

    The streaming flush (bobert_companion._SentenceFlushBuffer) never voices
    such a sentence early: it is said before the reply's actions run, and a
    reply with no token is not spoken at all (2026-10-02)."""
    if not text or not text.strip():
        return None
    ran = _ran_tokens(ran_actions)
    any_ran = bool(ran)
    passive_ok = not (_owner_asks_status(user_text)
                      or _owner_states(user_text))
    owner_words = _passive_words(_norm(user_text)) if passive_ok else set()
    for ack, ack_text, core in _segments(text):
        if ack == "done" and not any_ran:
            return ack_text
        if not core:
            continue
        got = _clause_completed_claim(core)
        tokens = _family_tokens(got[0]) if got else None
        if got is None and passive_ok:
            got = _passive_claim(core, owner_words)
            if got:
                tokens = _PASSIVE_TOKENS.get(got[0]) or _family_tokens(got[0])
        if got and not _grounded(tokens, ran, any_ran, got[1]):
            return got[1].strip()
    return None


def _calculate_claim(core, progressive, perfect, future, narration):
    """The "calculate" family's own reading of one clause (2026-10-02 review
    repair). ("calculate", phrase) for a COMPLETED-work claim ("I've run the
    calculations", "I crunched the numbers"); ("calculate_status", phrase) for
    a working line - "I'm running the numbers", "Let me calculate that", or a
    clause that is nothing but the gerund ("Calculating, sir."); else None.
    A clause-initial gerund or participle followed by more words is a noun or
    an adjective ("Computing power roughly doubles", "Calculated risk")."""
    m = perfect.search(core)
    if m:
        return "calculate", m.group(0)
    for rx in (progressive, future):
        m = rx.search(core)
        if m:
            return "calculate_status", m.group(0)
    m = narration.match(core)
    if m and _BARE_REST_RE.match(core[m.end():]):
        return "calculate_status", m.group(0)
    return None


def _ran_tokens(ran_actions: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for n in ran_actions or ():
        low = str(n).lower()
        tokens.add(low)
        tokens.update(t for t in re.split(r"[_\W]+", low) if t)
    return tokens


# A claim worded in one family that another family's action carries out
# (review 2026-10-02): "I've switched off the lamp" / "the lights have been
# switched on" is what smart_home_control does (the turn family's tokens),
# and "it has been sent to the printer" is print_document. A flagged reply is
# withheld, so a report of an action that ran must not be read as a claim.
# Grounding only - the lenient direction.
_GROUND_ALSO: dict[str, frozenset[str]] = {
    "switch": next(t for n, _g, _p, _b, t in _FAMILIES if n == "turn"),
    "send": frozenset({"print", "printer"}),
}


def _family_tokens(name: str) -> frozenset[str]:
    for fam, _rx, tokens in _COMPILED:
        if fam == name:
            return tokens | _GROUND_ALSO.get(name, frozenset())
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


def asks_owner(text: str) -> bool:
    """True when a REPLY asks the owner something: a question anywhere in it
    or an offer ("Shall I open it", "I'll open it if you'd like", "say the
    word") — the same offer table that keeps those out of the claim check.
    Used by core/turn_checker.py: a turn that waits on the owner is never
    retried."""
    t = _norm(text)
    return "?" in t or bool(_OFFER_RE.search(t))


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
    norm = _norm(text)
    # A completed calculation is JARVIS's own arithmetic, not an invented
    # action, when the reply carries its result or the owner asked something
    # with numbers in it (2026-10-02).
    shows_working = bool(_ARITH_RESULT_RE.search(norm) or _AMOUNT_RE.search(norm)
                         or _QUANT_ASK_RE.search(_norm(user_text)))
    acks: list[tuple[str, str]] = []      # (kind, phrase)
    content: list[str] = []               # substantive non-claim cores
    working_line = ""                     # a calculate_status phrase
    # Passive completion claims ("<app> has been closed") - see
    # _passive_claim. Not read when the owner's turn asks about state.
    passive_ok = not (_owner_asks_status(user_text)
                      or _owner_states(user_text))
    owner_words = _passive_words(_norm(user_text)) if passive_ok else set()
    for ack, ack_text, core in _segments(text):
        if ack:
            acks.append((ack, ack_text))
        if not core:
            continue
        claim = _clause_claim(core)
        if claim is None and passive_ok:
            passive = _passive_claim(core, owner_words)
            if passive:
                fam, phrase = passive
                tokens = _PASSIVE_TOKENS.get(fam) or _family_tokens(fam)
                # Grounded by an action of the same family, by any action for
                # a generic "done", or by an action named after the subject
                # ("the print job has been sent" after print_document ran).
                if not _grounded(tokens, ran, any_ran, phrase):
                    return phrase
                content.append(core)
                continue
        if claim and claim[0] == "calculate" and shows_working:
            claim = None
        if claim and claim[0] == "calculate_status":
            # The phrasebook's working line ("Calculating, sir.") is persona
            # flavour ahead of an answer; with nothing after it but promises
            # it is the claim. Decided once the whole reply has been read.
            if not any_ran and not working_line:
                working_line = claim[1].strip()
            continue
        if claim:
            fam, phrase = claim
            tokens = _family_tokens(fam)
            if not (tokens & ran or ("*" in tokens and any_ran)):
                return phrase.strip()
            content.append(core)     # a grounded summary of a real result
            continue
        if _substantive(core):
            content.append(core)
    if working_line and not any(not _PROMISE_RE.search(c) for c in content):
        return working_line
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
