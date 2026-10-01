"""core/yes_no.py — one classifier for every spoken yes/no answer.

JARVIS asks the owner yes/no questions in three places: the high-risk action
confirmation ("This will delete X. Shall I proceed?"), the autocorrect "did you
mean X or Y" pick, and the shutdown prompt ("overnight protocol first? Yes or
no."). Each router grew its own word list and its own matching rule, and they
drifted (2026-10-01 bug-hunt):

  * the confirmation gate tested a raw ``startswith`` against yes/confirm/do
    it/go ahead/proceed, so "Yeah." / "Sure." / "Okay." CANCELLED the action
    while "Yesterday we..." / "Do it later" / "Go ahead and cancel it" /
    "Confirmation number 5" CONFIRMED a delete / purchase / dangerous shell
    command;
  * the shutdown prompt compared the bare lower-cased text, so Whisper's
    punctuated "Yes." / "No." matched nothing (live miss 2026-09-30);
  * none of them dropped the wake word, so in wake-word mode — where every
    spoken reply MUST start with "Jarvis" — "Jarvis, yes." was never a yes.

``normalize`` and ``classify_reply`` are the single rule they now share — and
(2026-10-01 review) so do the older copies: the outbound draft gates
(core/draft_preview_gate.py, core/draft_confirm.py), which sent a draft on a
confirm word ANYWHERE in the reply ("Yeah, I saw it"), and the printer setup
wizard (skills/bambu_setup.py), whose raw startswith took "Yesterday..." /
"Right, so..." as a yes. Their own words ride on ``extra_yes`` / ``extra_no``.

classify_reply(text) -> "yes" | "no" | "other":

  * "no"    — the reply opens with a refusal ("No.", "Nope, cancel it", "Wait"),
              or opens like a yes but carries a hedge ("Do it later", "Go ahead
              and cancel it", "Yes, but wait"). A hedged yes is never a yes: the
              thing being confirmed can be destructive.
  * "yes"   — a SHORT clear yes. After a strong yes word ("yes", "yeah",
              "absolutely") or a strong lead ("go ahead", "do it") only filler,
              another yes / yes lead, or at most ONE other word may follow
              ("Yes, delete it", "Yes I am"). After a soft one ("Okay",
              "Sure") or a sentence-opener lead ("I am", "Please do") nothing
              but filler or another yes may follow ("Okay, what's the
              weather?" is NOT a yes — "okay" there is a discourse marker).
  * "other" — anything else: "Yesterday we...", "Confirmation number 5", a
              fresh command, and a sentence that merely STARTS with a yes word
              ("Yeah, I saw that movie last week.", "Yeah, I saw it.",
              "Absolutely, the game was great.") — 2026-10-01 review: the first cut let any hedge-free
              sentence after a strong yes word through, so the commonest way
              people start a sentence confirmed a queued delete. Never a yes;
              every caller treats it as "cancel / not an answer".

Words match WHOLE: "yesterday" is not "yes", "confirmation" is not "confirm".

Pure stdlib, no I/O, never raises — testable on the light-deps CI runner.
"""
from __future__ import annotations

import re

# Clear yes words: up to _MAX_OTHER_WORDS other words may follow ("Yes,
# delete it"); a longer sentence is "other" (see classify_reply).
STRONG_YES = frozenset({
    "yes", "yeah", "yep", "yup", "yea", "ya",
    "confirm", "confirmed", "proceed", "affirmative",
    "absolutely", "certainly", "definitely", "correct", "indeed",
})
# Soft yes words double as discourse markers ("Okay, so what time is it?"),
# so they count only when nothing but filler or another yes follows.
SOFT_YES = frozenset({"ok", "okay", "sure", "alright", "fine"})
YES_WORDS = STRONG_YES | SOFT_YES
# Multi-word yes openers, held to the strong rule ("go ahead and send it").
STRONG_LEADS = (("go", "ahead"), ("do", "it"), ("of", "course"))
# ...and to the soft rule: these also open ordinary sentences ("I am going
# out", "Please do the dishes"), so only filler may follow. "I'm sure." /
# "I am." answer the pushback "Are you certain?" (2026-10-01).
SOFT_LEADS = (("go", "for", "it"), ("please", "do"), ("i", "am", "sure"),
              ("i", "am", "certain"), ("im", "sure"), ("im", "certain"),
              ("i", "am"), ("sounds", "good"))
YES_LEADS = STRONG_LEADS + SOFT_LEADS
# The most "other" words a strong yes / strong lead may carry and still be a
# yes: "Yes, delete it" passes ("Yes I am" / "Yes, send it" are a yes plus a
# yes lead); "Yeah, I saw it" and "Yeah, I saw that movie last week" do not
# (2026-10-01 review).
_MAX_OTHER_WORDS = 1
# Openers that look like a yes and are not: sarcasm ("Yeah, right"), a
# filler ("Ya know..."), "Correct me if I'm wrong...". Always "other".
_NOT_YES_OPENERS = (("yeah", "right"), ("yea", "right"), ("ya", "right"),
                    ("yep", "right"), ("ya", "know"), ("correct", "me"))

# Words that open a refusal. ("dont" is "don't" after normalize.)
NO_WORDS = frozenset({
    "no", "nope", "nah", "not", "negative", "cancel", "stop", "abort",
    "never", "dont", "wait", "hold", "nevermind",
})
# A hedge ANYWHERE in a yes-shaped reply makes it a refusal: "do it later",
# "go ahead and cancel it", "yes but wait", "sure, not now".
HEDGES = NO_WORDS | frozenset({
    "not", "later", "after", "tomorrow", "tonight", "but", "except",
    "instead", "until",
})
# Affirmative idioms that contain a hedge word; removed before the hedge test
# so "Sure, why not." / "Yes, no problem." stay a yes.
_IDIOMS = (("why", "not"), ("no", "problem"), ("not", "a", "problem"),
           ("no", "worries"))
# Allowed after any yes word or yes lead without changing the answer.
FILLER = frozenset({
    "sir", "please", "jarvis", "now", "then", "thanks", "thank", "you",
    "thing", "so", "well", "oh", "um", "uh", "and", "lets", "go", "do",
    "it", "that", "right", "away", "ahead", "course", "of", "for",
    "certain",
})

# Leading wake word: "jarvis", "hey jarvis", "ok jarvis", "okay jarvis".
_WAKE_LEAD = (("hey", "jarvis"), ("ok", "jarvis"), ("okay", "jarvis"),
              ("jarvis",))
# Trailing address / politeness that never changes the answer.
_TRAILING = frozenset({"sir", "jarvis", "please"})


def normalize(text) -> str:
    """Lower-case words only, the wake word and trailing address dropped:
    'Jarvis, yes.' -> 'yes', 'No, thanks.' -> 'no thanks', "Don't." ->
    'dont', 'Shut-down' -> 'shut down'. A reply that is ONLY the wake word
    keeps it ('Jarvis.' -> 'jarvis'). Never raises."""
    try:
        s = str(text or "").lower().replace("’", "'")
        s = s.replace("'", "")                      # don't -> dont
        words = re.sub(r"[^\w\s]", " ", s).split()
        for lead in _WAKE_LEAD:
            n = len(lead)
            if tuple(words[:n]) == lead and len(words) > n:
                words = words[n:]
                break
        while len(words) > 1 and words[-1] in _TRAILING:
            words = words[:-1]
        return " ".join(words)
    except Exception:
        return ""


def _drop_idioms(words: list) -> list:
    out: list = []
    i = 0
    while i < len(words):
        for idiom in _IDIOMS:
            n = len(idiom)
            if tuple(words[i:i + n]) == idiom:
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out


def hedge_words(words) -> list:
    """The hedge words (HEDGES) in ``words`` — a list of normalized words —
    with the affirmative idioms ("why not", "no problem") dropped first. The
    shutdown prompt uses it on the words after its "no" (2026-10-01): "No,
    wait." is a cancel there, not a power-off. Never raises."""
    try:
        return [w for w in _drop_idioms(list(words or ())) if w in HEDGES]
    except Exception:
        return []


def _match_lead(words: list, leads) -> int:
    """Length of the longest lead in ``leads`` that ``words`` opens with, or
    0."""
    best = 0
    for lead in leads:
        n = len(lead)
        if n > best and tuple(words[:n]) == tuple(lead):
            best = n
    return best


def _other_words(rest: list, leads) -> int:
    """How many words of ``rest`` are not filler, a yes word or part of a
    yes lead ("Okay, go ahead" -> 0, "Yes, delete it" -> 1)."""
    n_other = 0
    i = 0
    while i < len(rest):
        n = _match_lead(rest[i:], leads)
        if n:
            i += n
            continue
        if rest[i] not in FILLER and rest[i] not in YES_WORDS:
            n_other += 1
        i += 1
    return n_other


def classify_reply(text, extra_yes=(), extra_no=()) -> str:
    """'yes' | 'no' | 'other' for a reply to a yes/no question (see the
    module docstring). Never raises: any error is 'other', which no caller
    ever treats as a yes.

    ``extra_yes``: a caller's own yes leads as word tuples (the draft gate's
    ("send",) / ("ship", "it"), the printer wizard's ("thats", "right")),
    held to the SOFT rule — only filler may follow. ``extra_no``: a caller's
    own refusal words ("wrong", "scrap"), counted like NO_WORDS at the
    start of the reply."""
    try:
        words = normalize(text).split()
        # A polite lead-in: "Please, go ahead." -> "go ahead" (but "Please
        # do." is itself a yes, and "Please." alone stays "other").
        if (len(words) > 1 and words[0] == "please"
                and tuple(words[:2]) != ("please", "do")):
            words = words[1:]
        if not words:
            return "other"
        extra_no = frozenset(extra_no or ())
        if words[0] in NO_WORDS or words[0] in extra_no:
            return "no"
        extra_yes = tuple(tuple(x) for x in (extra_yes or ()))
        soft_leads = SOFT_LEADS + extra_yes
        all_leads = STRONG_LEADS + soft_leads
        lead = _match_lead(words, all_leads)
        if lead:
            soft = _match_lead(words, soft_leads) == lead
        elif words[0] in YES_WORDS:
            lead, soft = 1, words[0] in SOFT_YES
        else:
            return "other"
        rest = _drop_idioms(words[lead:])
        if any(w in HEDGES or w in extra_no for w in rest):
            return "no"
        if any(tuple(words[:len(o)]) == o for o in _NOT_YES_OPENERS):
            return "other"
        n_other = _other_words(rest, all_leads)
        if n_other > (0 if soft else _MAX_OTHER_WORDS):
            return "other"
        return "yes"
    except Exception:
        return "other"
