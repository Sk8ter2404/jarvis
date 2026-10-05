"""core/name_suggest.py - "did you mean Claude?" for a misheard window name.

WHY THIS EXISTS (2026-10-05)
============================
Live 00:25:02 the speech model (Parakeet) heard "close everything except for
Claude" as "... except for Claw". close_all_windows_except found no "Claw"
window and, correctly, closed nothing - "everything except <nothing>" is
everything - but the answer stopped there: "I don't see a Claw window to
keep, sir, so I've closed nothing." The owner had to say the whole command
again. The same day the model heard "Skrillex" as "Skrillix", "Skrylix" and
"Skrillig".

This module is the pure half of the fix: given what was heard and the names
of the windows / apps actually open, the ONE name the owner most likely meant,
or None. A destructive caller (a bulk close) ASKS with it ("Did you mean
Claude?"); a harmless one (focus) may act on it and say so.

Matching (all on lower-case letters and digits only):
  * the heard name is compared with each candidate whole AND with each of its
    words ("Claude" inside "Claude Code"), so a suggestion is the form that
    matched, deduplicated case-insensitively;
  * a form is accepted when it is close by spelling (difflib ratio >= 0.8),
    or shares a 3+ letter start with a ratio >= 0.6 ("claw" / "claude"), or
    sounds the same by a crude consonant key with a ratio >= 0.5;
  * an exact match is never a suggestion, and a heard name that IS one of
    the candidates (or one of their words) gets none at all - the lookup's
    miss is then not a mishearing; names shorter than 3 letters are never
    matched, and when two different
    forms score within TIE_MARGIN of each other there is no suggestion: a
    guess between two windows is not ours to make.

Stdlib only, no I/O, never raises.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Optional

__all__ = ["compact", "phonetic_key", "score", "suggest"]

MIN_LEN = 3
STRONG_RATIO = 0.8
PREFIX_RATIO = 0.6
PREFIX_LEN = 3
PHONETIC_RATIO = 0.5
TIE_MARGIN = 0.05
# A candidate longer than this is a document / folder / page title, not a
# name anyone says; its words are still compared.
MAX_FORM_CHARS = 40

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'+#.]*")


def compact(s) -> str:
    """Lower-case letters and digits only: "Claude Code" -> "claudecode"."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def phonetic_key(word) -> str:
    """A crude sound key: vowels and soft letters dropped, look-alike
    consonants merged, repeats collapsed ("Skrillex", "Skrillix" and
    "Skrylix" all give "skrlks"). A leading vowel is kept as "a"."""
    w = re.sub(r"[^a-z]", "", str(word or "").lower())
    if not w:
        return ""
    w = (w.replace("ph", "f").replace("ck", "k").replace("qu", "kw")
         .replace("x", "ks"))
    out: list = []
    for ch in w:
        if ch in "aeiouyhw":
            c = ""
        elif ch in "ckqg":
            c = "k"
        elif ch in "sz":
            c = "s"
        elif ch in "dt":
            c = "t"
        elif ch in "bp":
            c = "p"
        elif ch in "fv":
            c = "f"
        elif ch in "mn":
            c = "n"
        else:
            c = ch
        if c and (not out or out[-1] != c):
            out.append(c)
    key = "".join(out)
    return ("a" + key) if w[0] in "aeiouy" else key


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def score(heard, form) -> float:
    """How well ``form`` explains ``heard`` (0.0 = not a suggestion). Both
    are compacted first; an exact match scores 0.0 (it is not a mishearing
    of anything)."""
    try:
        h, f = compact(heard), compact(form)
        if len(h) < MIN_LEN or len(f) < MIN_LEN or h == f:
            return 0.0
        r = SequenceMatcher(None, h, f).ratio()
        if r >= STRONG_RATIO:
            return r
        if _common_prefix(h, f) >= PREFIX_LEN and r >= PREFIX_RATIO:
            return r
        hk, fk = phonetic_key(heard), phonetic_key(form)
        if len(hk) >= 2 and hk == fk and r >= PHONETIC_RATIO:
            return r
    except Exception:
        return 0.0
    return 0.0


def _forms(candidate: str) -> list:
    """(form, display) pairs for one candidate name: the whole name (when
    short enough to be said) and each of its words of MIN_LEN+ letters."""
    out = []
    c = " ".join(str(candidate or "").split())
    if not c:
        return out
    if len(c) <= MAX_FORM_CHARS:
        out.append(c)
    words = [w.strip(".'") for w in _WORD_RE.findall(c)]
    if len(words) > 1:
        out.extend(w for w in words if len(compact(w)) >= MIN_LEN)
    return out


def suggest(heard, candidates: Iterable) -> Optional[str]:
    """The ONE name among ``candidates`` (or one of their words) that
    ``heard`` most likely was, else None (nothing close, or a tie between two
    different names). Never raises."""
    try:
        if not isinstance(heard, str):
            return None
        h = compact(heard)
        if len(h) < MIN_LEN:
            return None
        best: dict = {}      # casefolded form -> (score, display)
        for cand in candidates or ():
            if not isinstance(cand, str):
                continue
            for form in _forms(cand):
                if compact(form) == h:
                    # The heard name IS open: whatever failed to match it,
                    # another name is not the answer.
                    return None
                s = score(heard, form)
                if s <= 0.0:
                    continue
                key = compact(form)
                if key not in best or s > best[key][0]:
                    best[key] = (s, form)
        if not best:
            return None
        ranked = sorted(best.values(), key=lambda p: p[0], reverse=True)
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < TIE_MARGIN:
            # "Claude" (the app) and "Claude Code" (a terminal) both say
            # "claude" - one form, so not a tie. Two DIFFERENT forms this
            # close are a guess.
            return None
        return ranked[0][1]
    except Exception:
        return None
