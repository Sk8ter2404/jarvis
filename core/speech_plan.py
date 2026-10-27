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

  claim      a completed-work claim nothing run before this reply grounds
             (core.claim_validator.find_completed_claim, read-only);
  reading    a volume / battery / temperature figure nothing read before this
             reply grounds (core.claim_validator.find_ungrounded_reading);
  finding    "it appears / seems / looks like / I can see / is showing" - only
             while no look-up result has come back this turn (pending_lookup),
             so a follow-up round reading back a real result is untouched
             (10-26 20:06:02);
  inability  "I can't / I've no way to ... check / see / tell" while the held
             action - the one that checks - ran without failing;
  moot offer the head's LAST sentence offers or asks permission ("Shall I run
             the numbers, sir?") for the held action that then ran.

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

# First-person inability to look / check.
_INABLE_RE = re.compile(
    r"\b(?:i\s+(?:can'?t|cannot|can\s+not)|"
    r"i(?:'m|\s+am)\s+(?:unable|not\s+able)\s+to|"
    r"i(?:'ve|\s+have)\s+no\s+(?:way|means)\s+(?:to|of)|"
    r"i\s+(?:don'?t|do\s+not)\s+have\s+(?:a\s+|any\s+)?(?:way|means)\s+(?:to|of)|"
    r"(?:there'?s|there\s+is)\s+no\s+way\s+(?:for\s+me\s+)?to)"
    r"[^.?!]{0,80}?\b(?:check|see|tell|confirm|look|verify|know|access|"
    r"read|monitor|detect)\w*\b")

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
    if pending_lookup and _FINDING_RE.search(n):
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


def hold_head(head, *, results=(), pending_lookup: bool = True,
              user_text: str = "", ran_before: Iterable[str] = (),
              grounding_before: Iterable[str] = (), held_name: str = "",
              held_failed: Optional[bool] = None, whole: bool = False):
    """(kept_text, dropped) for the prose ``head`` spoken before a held
    result.

    results           this reply's (name, result, informative) entries.
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
                                 user_text=user_text, ran_before=ran_before,
                                 grounding_before=grounding_before,
                                 held_ok=held_ok, whole=whole)
            if (why is None and i == last and held_ok and not whole
                    and _is_offer(_TAG_RE.sub(" ", s))):
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
