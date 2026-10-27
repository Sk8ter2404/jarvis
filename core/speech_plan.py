"""Speech plan: what a reply may say BEFORE its first held action token.

Head hold (v3 W02, 2026-10-27). A reply is written in one pass, before any of
its actions run. parse_and_run_actions already holds the prose AFTER the
first token whose result the model has not seen (result hold), but the prose
BEFORE it - the head - was always spoken. Live:

  10-26 20:05:48  "I've taken a look at the screen, sir - it appears the video
                  hasn't loaded ..." [ACTION: local_describe_screen]  (the
                  vision result then said otherwise)
  10-26 20:05:40  "I'm afraid I've no way to check if the video is actually
                  playing without running a scan, sir." [ACTION: screenshot]
  10-19 10:46:11  "I'm afraid I can't check that directly ..."
                  [ACTION: which_mic_is_active]
  10-19 19:41:05  "Shall I run the numbers, sir?" [ACTION: system_pulse]

hold_head() judges each head sentence. While a held result is still to come
(the follow-up round or the verbatim read-out WILL speak), a head may say "I'm
doing it" - an acknowledgement, progress, a question to the owner. It may not
say what happened or what the result is. A sentence is DROPPED when it is:

  claim      a completed-work claim nothing that ran grounds - this turn
             before the reply, or this reply's other actions (they have all
             run by the time the head is spoken; the held one is excluded)
             (core.claim_validator.find_completed_claim, read-only);
  reading    a volume / battery / temperature figure nothing read grounds
             (the same span; core.claim_validator.find_ungrounded_reading);
  finding    "it appears / seems / looks like / I can see / is showing" - not
             inside a "let me see what / check whether ..." clause, and only
             while no look-up result has come back this turn (pending_lookup),
             so a follow-up round reading back a real result is untouched
             (10-26 20:06:02);
  inability  "I can't / I've no way to ... check / see / tell" while the held
             action - the one that checks - ran without failing;
  moot offer the head's LAST sentence offers or asks permission ("Shall I run
             the numbers, sir?") for the held action that then ran: it names
             the action or offers a generic look / run, with no "also /
             another / else / next" (an offer of another step stays).

Everything else is kept. With ``whole=True`` (an AMBIGUOUS / UNCLEAR
autocorrect result: nothing ran, the owner is asked) only claim, reading and
finding apply and every question stays.

Pure: stdlib + core.claim_validator (imported, never edited). Never raises:
on any fault the head comes back unchanged with no drops.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from core import claim_validator as _cv
from core.failure_markers import FAILURE_MARKERS as _FAILURE_MARKERS

__all__ = ("hold_head", "split_sentences", "judge_sentence")

# One sentence: up to terminal punctuation followed by whitespace or the end
# (the same shape bobert_companion._ACK_SENTENCE_RE uses).
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.S)
# Any bracket tag ([intent:x], [mood:x], [wry], a stray [ACTION: ...]).
_TAG_RE = re.compile(r"\[[^\]\n]*\]")
# The tags that lead the head: kept whatever is dropped, so the TTS preset the
# reply chose still applies to what is said.
_LEAD_TAGS_RE = re.compile(r"^(?:\s*\[[^\]\n]*\])+\s*")

# Findings: the reply reports what something LOOKS like.
_FINDING_RE = re.compile(
    r"\b(?:it\s+(?:appears|seems|looks)|it\s+would\s+(?:appear|seem)|"
    r"appears\s+to|seems\s+to|looks\s+like|i\s+can\s+see|"
    r"i\s+see\s+(?:that|the)|is\s+showing)\b")

# A finding phrase inside a subordinate check clause ("let me see what the
# screen is showing", "checking whether it appears to run") is progress, not
# a finding: the text before the match opens such a clause and no clause
# break comes between (review 2026-10-27).
_SUBORDINATE_TAIL_RE = re.compile(
    r"\b(?:if|whether|what|how|where|which)\b"
    r"(?:(?!\s[-–—]\s)[^.?!,;:–—])*$")
# "I can see to that / about it / if ..." is a promise or a check, not a look.
_CAN_SEE_TO_RE = re.compile(r"i\s+can\s+see\s+(?:to|about|if|whether)\b")

# First-person inability to look / check.
_INABLE_RE = re.compile(
    r"\b(?:i\s+(?:can'?t|cannot|can\s+not)|"
    r"i(?:'m|\s+am)\s+(?:unable|not\s+able)\s+to|"
    r"i(?:'ve|\s+have)\s+no\s+(?:way|means)\s+(?:to|of)|"
    r"i\s+(?:don'?t|do\s+not)\s+have\s+(?:a\s+|any\s+)?(?:way|means)\s+(?:to|of)|"
    r"(?:there'?s|there\s+is)\s+no\s+way\s+(?:for\s+me\s+)?to)"
    r"[^.?!]{0,80}?\b(?:check|see|tell|confirm|look|verify|know|access|"
    r"read|monitor|detect)\w*\b")

# Moot offer: the last head sentence offers the held action only when it is
# ABOUT that action - it names a word of the action's name, or offers a
# generic look / run at "it" / "the numbers" - and carries no marker of a
# further step (review 2026-10-27: "I can also check the weather?" before a
# screen look is a real question and stays).
_OFFER_OTHER_RE = re.compile(
    r"\b(?:also|too|another|else|afterwards?|as\s+well|next|then|"
    r"in\s+the\s+meantime|meanwhile|while\s+i'?m\s+at\s+it)\b")
_OFFER_GENERIC_RE = re.compile(
    r"\b(?:run|check|look|see|scan|pull|fetch|get|have|take|do|give)\b"
    r"(?:\s+(?:a|the|it|that|this|them|those|up|over|in|into|on|at|for|"
    r"quick|quickly|proper|full|closer|you|your|figures?|numbers?|"
    r"stats?|statistics|diagnostics?|readings?|look|glance|peek|scan|"
    r"check|sir|now|again|here)\b){0,5}"
    r"(?=\s*(?:,|\?|$|\bsir\b|\bfor\s+you\b|\bnow\b))")
_OFFER_NUMBERS_RE = re.compile(
    r"\b(?:the\s+)?(?:numbers|figures|stats|statistics|diagnostics?|"
    r"readings?)\b")
_NAME_STOP = frozenset({
    "get", "set", "show", "open", "run", "check", "info", "list", "the",
    "now", "off", "local", "status", "what", "which", "how", "is", "my",
})


def _offer_is_about(core: str, held: str) -> bool:
    """True when the offer sentence ``core`` is about the held action
    ``held`` (see _OFFER_OTHER_RE). Never raises."""
    try:
        n = _norm(core)
        if _OFFER_OTHER_RE.search(n):
            return False
        words = set(re.findall(r"[a-z]+", n))
        for tok in re.split(r"[^a-z]+", held.lower()):
            if len(tok) < 3 or tok in _NAME_STOP:
                continue
            if any(w == tok or (len(tok) >= 4 and w.startswith(tok))
                   for w in words):
                return True
        if _OFFER_NUMBERS_RE.search(n):
            return True
        return bool(_OFFER_GENERIC_RE.search(n))
    except Exception:
        return False


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("’", "'")
                  .replace("‘", "'")).strip().lower()


def split_sentences(text: str) -> list:
    """``text`` as its sentences, in order, each stripped."""
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(str(text or ""))
            if m.group(0).strip()]


def _is_question(core: str) -> bool:
    return core.rstrip().endswith("?")


def _is_offer(core: str) -> bool:
    n = _norm(core)
    return _is_question(n) and bool(_cv._OFFER_RE.search(n))


def _is_finding(n: str) -> bool:
    """True when the normalised sentence ``n`` reports what something looks
    like: a _FINDING_RE phrase that is not inside a subordinate check clause
    ("let me see what it is showing") nor "i can see to that". Never
    raises."""
    try:
        for m in _FINDING_RE.finditer(n):
            if _CAN_SEE_TO_RE.match(n, m.start()):
                continue
            if _SUBORDINATE_TAIL_RE.search(n[:m.start()]):
                continue
            return True
    except Exception:
        return False
    return False


def judge_sentence(sentence: str, *, pending_lookup: bool = True,
                   user_text: str = "", ran_before: Iterable[str] = (),
                   grounding_before: Iterable[str] = (),
                   held_ok: bool = False, whole: bool = False
                   ) -> Optional[str]:
    """Why one head sentence must not be spoken ("claim" / "reading" /
    "finding" / "inability"), or None to keep it. The moot-offer rule needs
    the sentence's place in the head, so hold_head applies it."""
    core = _TAG_RE.sub(" ", str(sentence or ""))
    core = re.sub(r"\s+", " ", core).strip()
    if not core:
        return None
    if _cv.find_completed_claim(core, ran_actions=list(ran_before or ()),
                                user_text=user_text or ""):
        return "claim"
    if _cv.find_ungrounded_reading(core, grounding=list(grounding_before or ()),
                                   user_text=user_text or ""):
        return "reading"
    if _is_question(core):
        return None             # a question to the owner is always kept
    n = _norm(core)
    if pending_lookup and _is_finding(n):
        return "finding"
    if held_ok and not whole and _INABLE_RE.search(n):
        return "inability"
    return None


def _held_failed(results, held_name: str) -> bool:
    for entry in results or ():
        try:
            name, result = entry[0], entry[1]
        except Exception:
            continue
        if str(name or "").strip().lower() != held_name:
            continue
        low = str(result or "").lower()
        return any(str(mk).lower() in low for mk in _FAILURE_MARKERS)
    return False


_DEFERRED_PREFIXES = ("⚠  PUSHBACK:", "⚠  REQUIRES CONFIRMATION:",
                      "⚠  AMBIGUOUS:", "⚠  UNCLEAR:")


def _with_reply_results(results, held: str, ran_before, grounding_before,
                        failed_names=()):
    """(ran, grounding): ``ran_before`` / ``grounding_before`` plus this
    reply's own results other than the held action. A result that failed
    (failure markers, or named in ``failed_names``) adds no ran name, as the
    turn ledger (_note_turn_action_ran) does; a deferred one (pushback /
    confirmation / ambiguity: nothing ran) and a synthetic "_" one add
    nothing. Raises on unreadable ``results`` (hold_head then fails open)."""
    ran = list(ran_before or ())
    grounding = list(grounding_before or ())
    failed = {str(n or "").strip().lower() for n in failed_names or ()}
    for entry in results or ():
        name = str(entry[0] or "").strip().lower()
        result = entry[1]
        if not name or name == held or name.startswith("_"):
            continue
        if isinstance(result, str) and result.startswith(_DEFERRED_PREFIXES):
            continue
        low = str(result or "").lower()
        grounding.append(str(result or "")[:2000])
        if name in failed or any(str(mk).lower() in low
                                 for mk in _FAILURE_MARKERS):
            continue
        ran.append(name)
    return ran, grounding


def hold_head(head, *, results=(), pending_lookup: bool = True,
              user_text: str = "", ran_before: Iterable[str] = (),
              grounding_before: Iterable[str] = (), held_name: str = "",
              held_failed: Optional[bool] = None, whole: bool = False,
              failed_names: Iterable[str] = ()):
    """(kept_text, dropped) for the prose ``head`` spoken before a held
    result.

    results           this reply's (name, result, informative) entries. Every
                      one but the held action has run by the time the head
                      is spoken, so its name grounds a claim and its result a
                      reading (review 2026-10-27: "I've opened Chrome, sir."
                      before [ACTION: open_app] [ACTION: see_screen] is true).
    failed_names      names in ``results`` the caller knows failed or refused
                      (beyond the failure markers); they ground no claim.
    pending_lookup    True while no look-up result has come back this turn
                      (findings can only be guesses); False in a follow-up
                      round that is reading one back.
    user_text         the owner's words this turn.
    ran_before        actions that ran this turn BEFORE this reply.
    grounding_before  what those actions said (readings ground on it).
    held_name         the held action (whose result will speak); "" for none.
    held_failed       whether it failed / refused; None = judge ``results``.
    whole             an AMBIGUOUS / UNCLEAR deferral: claim, reading and
                      finding only; questions stay.

    ``dropped`` is a list of (sentence, why). With nothing dropped the head
    comes back unchanged (the same string). Never raises."""
    try:
        text = str(head or "")
        if not text.strip():
            return head, []
        held = str(held_name or "").strip().lower()
        if held_failed is None:
            held_failed = _held_failed(results, held) if held else True
        held_ok = bool(held) and not held_failed
        ran_eff, grounding_eff = _with_reply_results(
            results, held, ran_before, grounding_before, failed_names)
        m = _LEAD_TAGS_RE.match(text)
        lead = m.group(0).strip() if m else ""
        body = text[m.end():] if m else text
        sentences = split_sentences(body)
        if not sentences:
            return head, []
        kept, dropped = [], []
        last = len(sentences) - 1
        for i, s in enumerate(sentences):
            why = judge_sentence(s, pending_lookup=pending_lookup,
                                 user_text=user_text, ran_before=ran_eff,
                                 grounding_before=grounding_eff,
                                 held_ok=held_ok, whole=whole)
            if (why is None and i == last and held_ok and not whole
                    and _is_offer(_TAG_RE.sub(" ", s))
                    and _offer_is_about(_TAG_RE.sub(" ", s), held)):
                why = "moot offer"
            if why:
                dropped.append((s, why))
            else:
                kept.append(s)
        if not dropped:
            return head, []
        if not kept:
            return "", dropped
        return (" ".join(([lead] if lead else []) + kept)).strip(), dropped
    except Exception:
        return head, []
