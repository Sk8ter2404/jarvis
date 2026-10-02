"""core/offer_reply.py — a yes to JARVIS's own offer carries the offer out.

Live 2026-10-02 (14:48): a turn ended on JARVIS offering to move an app's
window to another monitor ("Should I ... move the existing window to the top
monitor?"). The owner answered with a bare "Jarvis, yes." Nothing in the
monolith knew that the yes answered an offer: the high-risk confirmation queue
(``_pending_confirmation``) was empty, so the yes went to the brain like any
other turn, with nothing saying what it answered, and the brain answered with a
quip and ran nothing. (That action is documented in an always-on prompt
section; one documented in a routed section would not even have reached the
model, because "yes" routes no section.)

This module is the pure half of the fix:

  * ``offer_text(reply)`` — the offer a reply ENDS with, when it is a yes/no
    offer that names a concrete action ("Shall I put it on the top monitor?",
    "Would you like me to read them to you?"). A wh-question, a choice ("the
    left or the right?") and an offer of nothing in particular ("Would you
    like to hear more?") are not offers a yes can carry out.
  * ``OpenOffer`` — the one offer the last finished turn ended on. Every owner
    turn TAKES it (an offer is answered or ignored by the very next turn), and
    a take only accepts a clear, prompt yes: the shared yes/no classifier's
    "yes" (core.yes_no.classify_reply — the wake word is dropped, a hedged
    yes such as "yes, but later" is a no), no other owner turn in between,
    and no later than ``ttl_s`` after the offer was made.
  * ``directive(offer)`` — the per-turn note that tells the brain the owner
    said yes to that offer and must now emit its action token. The monolith
    also routes the turn's prompt sections on the offer's words, so an action
    documented in a routed section reaches the model too.

The action still goes through the normal dispatcher, so every gate applies
unchanged: a confirm-keyword action still asks for its own yes, and a
self-terminating one still needs the owner's own words.

Pure stdlib (plus core.yes_no), no I/O, injectable clock, never raises —
testable on the light-deps CI runner.
"""
from __future__ import annotations

import re
import time

from core import yes_no as _yes_no

# Bracketed tags and action tokens ("[intent:dry_wit]", "[ACTION: x, y]").
_TAG_RE = re.compile(r"\[[^\]\n]*\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

# The speaker offering to DO something. Matched case-insensitively against
# one sentence; the action verb must come AFTER the match.
_OFFER_LEAD_RE = re.compile(
    r"\b(?:shall|should|may|can|could)\s+i\b"
    r"|\bshall\s+we\b"
    r"|\b(?:want|like)\s+(?:for\s+)?me\s+to\b"
    r"|\bwould\s+you\s+like\s+(?:for\s+)?(?:me\s+)?to\b"
    r"|\bdo\s+you\s+want\s+(?:me\s+)?to\b",
    re.IGNORECASE)
# The statement form: "I can move it there if you'd like." / "If you'd like,
# I'll read them to you."
_I_CAN_RE = re.compile(r"\bi\s+(?:can|could|will)\b|\bi'll\b", re.IGNORECASE)
_IF_YOU_LIKE_RE = re.compile(
    r"\bif\s+you(?:'d|\s+would)?\s+(?:like|prefer|want)\b", re.IGNORECASE)

# Verbs that name something JARVIS can do. Base forms; compared with the
# words after the offer lead.
ACTION_VERBS = frozenset("""
    add arm attempt book bring brighten call cancel capture check clear click
    close connect continue convert copy create decrease delete dim disable
    disarm disconnect display draft email empty enable end fetch find finish
    focus forward give hide increase install kill launch load lock log look
    lower make mark maximise maximize message minimise minimize move mute
    navigate note open pair pause pin play print pull put queue raise read
    reboot record refresh relaunch reload remind remove reopen repeat replay
    reply reset resize restart restore resume retry run save schedule scroll
    search send set share show shuffle shut skip snap start stop summarise
    summarize swap switch sync take text toggle translate try turn type unlock
    unmute unpin update write
""".split())

# Words that may stand between the offer lead and its verb ("Shall I ALSO
# mute it?", "Would you like me to GO AHEAD AND close it?", "Shall I, SIR,
# open it?"). The verb must be the FIRST other word (review 2026-10-02): JARVIS's
# dry asides carry an action verb in a clause about something else ("Should I
# be worried that you keep asking me to OPEN it?", "Can I just say that was a
# bold MOVE?"), and a joking yes to one told the brain to run it.
_LEAD_BRIDGE = frozenset({
    "also", "just", "go", "ahead", "and", "now", "then", "quickly", "simply",
    "first", "perhaps", "maybe", "still", "kindly", "instead", "sir",
    "please", "right", "away",
    # "Shall I have a quick look?"
    "have", "a", "quick",
})
# A verb that opens a rhetorical aside, not an action: "Should I take it
# personally?", "Can I remind you, sir, that ...?".
_RHETORICAL_RE = re.compile(
    r"\btake\s+(?:it|that|this)\s+personally\b"
    r"|\bremind\s+you\b(?:\s*,?\s*sir\s*,?)?\s+that\b",
    re.IGNORECASE)
# "Shall I do that?" points back at an earlier sentence of the same reply.
_DO_BACKREF_RE = re.compile(
    r"\b(?:do|go\s+ahead\s+with)\s+(?:that|it|so|this)\b", re.IGNORECASE)
# A question that asks for information, not permission.
_WH_WORDS = frozenset({"what", "which", "where", "when", "who", "whom",
                       "whose", "why", "how"})
_LEADING_FILLER = frozenset({"sir", "and", "so", "well", "also", "then",
                             "now", "oh", "right", "very", "good", "certainly"})
# Longest offer text kept (and handed to the brain).
MAX_OFFER_CHARS = 240


def _words(text: str) -> list:
    return re.findall(r"[a-z']+", str(text or "").lower().replace("’", "'"))


def _leads(sentence: str) -> list:
    """The offer-lead matches of ``sentence``: the question leads, and for
    the "... if you'd like" statement form its "I can" / "I'll"."""
    leads = list(_OFFER_LEAD_RE.finditer(sentence))
    if _IF_YOU_LIKE_RE.search(sentence):
        leads += list(_I_CAN_RE.finditer(sentence))
    return leads


def _verbs_after_lead(sentence: str) -> bool:
    """True when ``sentence`` carries an offer lead whose first word after
    it (past a short bridge, _LEAD_BRIDGE) is an action verb, and is not a
    rhetorical aside (_RHETORICAL_RE)."""
    if _RHETORICAL_RE.search(sentence):
        return False
    for m in _leads(sentence):
        for w in _words(sentence[m.end():]):
            if w in _LEAD_BRIDGE:
                continue
            if w in ACTION_VERBS:
                return True
            break
    return False


def _is_choice_or_wh(sentence: str) -> bool:
    """A choice question ("left or right?") or a wh-question: a bare yes
    answers neither."""
    words = _words(sentence)
    if "or" in words:
        return True
    lead = [w for w in words if w not in _LEADING_FILLER]
    return bool(lead) and lead[0] in _WH_WORDS


def _is_offer_sentence(sentence: str) -> bool:
    """A yes/no offer question ("Shall I ...?"), or the statement form
    ("I can ... if you'd like.")."""
    s = sentence.strip()
    if not s or _is_choice_or_wh(s):
        return False
    if _OFFER_LEAD_RE.search(s) and s.endswith("?"):
        return True
    return bool(_IF_YOU_LIKE_RE.search(s) and _I_CAN_RE.search(s))


def offer_text(reply) -> str:
    """The offer ``reply`` ENDS with, or "" (see the module docstring).

    Tags and [ACTION: ...] tokens are removed first. The last sentence must
    be an offer whose words after the lead name an action; "Shall I do
    that?" takes the sentence before it along when that one names the
    action. Never raises."""
    try:
        prose = " ".join(_TAG_RE.sub(" ", str(reply or "")).replace(
            "’", "'").split())
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(prose)
                     if s.strip()]
        if not sentences:
            return ""
        last = sentences[-1]
        if not _is_offer_sentence(last):
            return ""
        if _verbs_after_lead(last):
            return last[:MAX_OFFER_CHARS]
        if len(sentences) >= 2 and _DO_BACKREF_RE.search(last):
            prev = sentences[-2]
            if any(w in ACTION_VERBS for w in _words(prev)):
                return (prev + " " + last)[:MAX_OFFER_CHARS]
        return ""
    except Exception:
        return ""


def directive(offer: str) -> str:
    """The per-turn note for the brain when the owner said yes to ``offer``
    (the leading blank line matches the other per-turn addenda)."""
    offer = " ".join(str(offer or "").split())[:MAX_OFFER_CHARS]
    return ("\n\nOFFER ACCEPTED: your previous reply ended with this offer to "
            f"sir: \"{offer}\" He has just answered yes. Carry out exactly "
            "that offer now: emit its [ACTION: ...] token in this reply, with "
            "the target and arguments the conversation above names. Do not "
            "ask again, and do not answer with only an acknowledgement.")


class OpenOffer:
    """The offer the last finished turn ended on, if any.

    ``note`` records it when the turn's chain ends (or clears the slot when
    the turn ended on anything else); ``take`` is called once per owner turn
    and always empties the slot. ``ttl_s``: how long after the offer a yes
    still answers it. ``clock``: the monolith's time.monotonic(), the clock
    of its owner-turn stamps."""

    def __init__(self, ttl_s: float, clock=None) -> None:
        try:
            self.ttl_s = max(0.0, float(ttl_s))
        except (TypeError, ValueError):
            self.ttl_s = 0.0             # a malformed TTL accepts nothing
        # Looked up at call time, so a test's patched time.monotonic holds.
        self._clock = clock or (lambda: time.monotonic())
        self._offer = ""
        self._at = 0.0

    def note(self, reply) -> str:
        """Record the offer ``reply`` ends with (offer_text) as the open
        offer, or clear the slot when it ends with none. Returns the offer.
        Never raises."""
        try:
            offer = offer_text(reply)
            self._offer = offer
            self._at = float(self._clock()) if offer else 0.0
            return offer
        except Exception:
            self.clear()
            return ""

    def clear(self) -> None:
        self._offer = ""
        self._at = 0.0

    def peek(self) -> str:
        return self._offer

    def take(self, answer, *, since: float = 0.0) -> "tuple[str, str]":
        """``(outcome, offer)`` for this owner turn ``answer``, and the slot
        is emptied whatever the outcome. ``since``: when the owner turn
        BEFORE this one started (the monolith's _prev_owner_turn_at) - an
        offer older than that was made before another owner turn and is
        stale. Outcomes:
          "none"     no offer was open
          "stale"    another owner turn came between the offer and this one
          "expired"  this turn came more than ttl_s after the offer
          "yes"      a clear yes: carry ``offer`` out
          "no"       a refusal or a hedged yes ("yes, but later")
          "other"    anything else: an unrelated turn
        Never raises (a fault is "other")."""
        offer, at = self._offer, self._at
        self.clear()
        try:
            if not offer:
                return "none", ""
            if since and at <= float(since):
                return "stale", offer
            if float(self._clock()) - at > self.ttl_s:
                return "expired", offer
            verdict = _yes_no.classify_reply(answer)
            if verdict in ("yes", "no"):
                return verdict, offer
            return "other", offer
        except Exception:
            return "other", offer
