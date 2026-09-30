"""core/proactive_guard.py — may this spontaneous remark be spoken, and now?

Pure checks behind bobert_companion's idle remarks (should_be_proactive ->
_do_proactive_turn -> generate_proactive_comment). Stdlib only, no I/O; the
monolith supplies the clock, the history and the state.

THE LIVE INCIDENTS (session_2026-09-29_22-06-02.log and
session_2026-09-30_09-48-41.log):
  * The model copied a persona-pool example VERBATIM, time of day included:
    "You seem rather determined this evening, sir." (the 'observation' bucket
    of the phrasebook) was the proactive remark at 22:43, 22:49, 23:17, 23:36,
    00:02, 07:54, 08:07, 08:14 — and again at 09:59 the next morning, when it
    was ten in the morning. The proactive prompt is the same every time, so
    the local model answers it the same way every time.
  * Nothing remembered what had already been said, so the same line played
    eight times across one night.
  * Nothing backed off: an unanswered remark was followed by another as soon
    as the silence window allowed (every 3-6 minutes).

check_remark() is the text gate (A1 + A2): a remark that names a time of day
the local clock contradicts, copies a persona example, or repeats a recent
remark is dropped. rate_verdict() is the pacing gate (A4): an unanswered
remark buys a longer wait, and a second one buys silence until the owner
speaks again.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

# ── time of day ──────────────────────────────────────────────────────────────
# Each phrase names a part of the day; it is right only at the hours below
# (local clock, inclusive, with an hour of slack at each edge so "this
# evening" at 00:30 or "this morning" at 12:40 is not a contradiction). A
# proactive remark is optional, so the windows err toward dropping.
_DAY_PART_HOURS: dict[str, frozenset] = {
    "morning":   frozenset(range(4, 13)),                        # 04-12
    "afternoon": frozenset(range(11, 19)),                       # 11-18
    "evening":   frozenset(list(range(16, 24)) + [0]),           # 16-00
    "tonight":   frozenset(list(range(16, 24)) + [0, 1, 2, 3]),  # 16-03
    "late":      frozenset(list(range(21, 24)) + list(range(0, 6))),  # 21-05
}

_TIME_PHRASES: tuple = (
    ("this morning",   re.compile(r"\bthis morning\b"),                 "morning"),
    ("good morning",   re.compile(r"\bgood morning\b"),                 "morning"),
    ("this afternoon", re.compile(r"\bthis afternoon\b"),               "afternoon"),
    ("good afternoon", re.compile(r"\bgood afternoon\b"),               "afternoon"),
    ("this evening",   re.compile(r"\bthis evening\b"),                 "evening"),
    ("good evening",   re.compile(r"\bgood evening\b"),                 "evening"),
    ("tonight",        re.compile(r"\btonight\b"),                      "tonight"),
    ("late night",     re.compile(r"\blate[\s-]+night\b"),              "late"),
    ("at this hour",   re.compile(r"\bat this (?:late |ungodly )?hour\b"), "late"),
    ("this late",      re.compile(r"\b(?:this|so) late\b"),             "late"),
    ("past midnight",  re.compile(r"\bpast midnight\b"),                "late"),
)

_TAG_RE = re.compile(r"\[[^\]]*\]")


def day_part(hour: int) -> str:
    """'morning' / 'afternoon' / 'evening' / 'night' for a local hour, the
    label the proactive prompt is given. Never raises."""
    try:
        h = int(hour) % 24
    except Exception:
        return "day"
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


def _plain(text) -> str:
    """Lower case, [tags] removed, curly apostrophes straightened."""
    s = _TAG_RE.sub(" ", str(text or ""))
    return s.replace("’", "'").replace("‘", "'").lower()


def time_phrases(text) -> list:
    """Every time-of-day phrase the text states, as (phrase, day part)."""
    s = _plain(text)
    return [(name, part) for name, rx, part in _TIME_PHRASES if rx.search(s)]


def time_of_day_conflict(text, hour: int) -> str:
    """The first time-of-day phrase in ``text`` that the local ``hour``
    contradicts ("this evening" at 10 AM), or "". Never raises."""
    try:
        h = int(hour) % 24
        for name, part in time_phrases(text):
            if h not in _DAY_PART_HOURS[part]:
                return name
        return ""
    except Exception:
        return ""


# ── text similarity ──────────────────────────────────────────────────────────
COPY_RATIO     = 0.8    # token-sequence similarity = near-verbatim copy
REPEAT_RATIO   = 0.75   # ... = a near-repeat of a recent remark
REPEAT_JACCARD = 0.7    # word-set overlap = a near-repeat of a recent remark
MIN_NEAR_WORDS = 3      # below this only an exact match counts

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# Words that carry no content for "is this the same remark": the address
# and bare articles/conjunctions.
_FILLER = frozenset({"sir", "the", "a", "an", "and", "so"})


def _tokens(text) -> tuple:
    """Content words, time-of-day phrases removed: 'You seem rather determined
    this evening, sir.' and 'You seem rather determined, sir.' both give
    ('you', 'seem', 'rather', 'determined')."""
    s = _plain(text)
    for _name, rx, _part in _TIME_PHRASES:
        s = rx.sub(" ", s)
    return tuple(w for w in _WORD_RE.findall(s) if w not in _FILLER)


def _similar(a: tuple, b: tuple, ratio: float, jaccard: float | None) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) < MIN_NEAR_WORDS or len(b) < MIN_NEAR_WORDS:
        return False
    if SequenceMatcher(None, a, b, autojunk=False).ratio() >= ratio:
        return True
    if jaccard is not None:
        sa, sb = set(a), set(b)
        if len(sa & sb) / len(sa | sb) >= jaccard:
            return True
    return False


# ── the persona pool ─────────────────────────────────────────────────────────
# Retired examples stay in the pool so a model that learned them (or a stale
# prompt) is still caught. The live offender of 2026-09-29/30:
RETIRED_EXAMPLES = (
    "You seem rather determined this evening, sir.",
    "You've been at this all afternoon, sir.",
)

# A capitalised single-quoted sentence in the persona prompt: 'Very good,
# sir.' — an inner apostrophe is one followed by a letter (I'm, that's).
_QUOTED_RE = re.compile(r"(?<![A-Za-z])'([A-Z](?:[^'\n]|'(?=[a-z]))*)'(?![A-Za-z])")


def quoted_examples(prompt_text) -> list:
    """The capitalised single-quoted example lines in a prompt body."""
    try:
        return [m.group(1).strip() for m in _QUOTED_RE.finditer(str(prompt_text or ""))]
    except Exception:
        return []


@lru_cache(maxsize=1)
def persona_pool() -> tuple:
    """Every persona example line the model is shown: the MCU phrasebook, the
    signature-opener pool, the quoted examples in the base persona prompt,
    plus RETIRED_EXAMPLES. Imported lazily, each source optional."""
    lines: list = list(RETIRED_EXAMPLES)
    try:
        import mcu_phrases
        for bucket in mcu_phrases.MCU_PHRASES.values():
            lines.extend(bucket)
    except Exception:
        pass
    try:
        from core import persona
        lines.extend(persona.JARVIS_SIGNATURE_PHRASES)
    except Exception:
        pass
    try:
        from core import prompts
        lines.extend(quoted_examples(prompts.BASE_SYSTEM_PROMPT))
    except Exception:
        pass
    seen: set = set()
    out: list = []
    for line in lines:
        if isinstance(line, str) and line and line not in seen:
            seen.add(line)
            out.append(line)
    return tuple(out)


def persona_copy(text, pool=None) -> str:
    """The persona example ``text`` copies verbatim or near-verbatim (time of
    day and 'sir' ignored), or "". ``pool`` defaults to persona_pool()."""
    try:
        t = _tokens(text)
        if not t:
            return ""
        for example in (persona_pool() if pool is None else pool):
            if _similar(t, _tokens(example), COPY_RATIO, None):
                return example
        return ""
    except Exception:
        return ""


def repeats_recent(text, recent) -> str:
    """The recent remark ``text`` repeats or near-repeats (normalised text,
    or high token overlap), or ""."""
    try:
        t = _tokens(text)
        if not t:
            return ""
        for prior in recent or ():
            if _similar(t, _tokens(prior), REPEAT_RATIO, REPEAT_JACCARD):
                return str(prior)
        return ""
    except Exception:
        return ""


def check_remark(text, *, hour: int, recent=(), pool=None) -> tuple:
    """(ok, reason) for one generated proactive remark at local ``hour``.

    Dropped (ok False) when it states a time of day the clock contradicts,
    copies a persona example, or repeats one of ``recent``. The reason names
    the rule (and the phrase/example it matched); it is for the log. An
    internal error drops the remark (a remark is optional, silence is safe)."""
    try:
        if not _tokens(text):
            return (False, "empty")
        bad = time_of_day_conflict(text, hour)
        if bad:
            return (False, f"says '{bad}' at {int(hour) % 24:02d}:00, "
                           f"{day_part(hour)}")
        copied = persona_copy(text, pool)
        if copied:
            return (False, f"copies the persona example '{copied}'")
        prior = repeats_recent(text, recent)
        if prior:
            return (False, "repeats a recent remark")
        return (True, "")
    except Exception as e:
        return (False, f"check failed: {type(e).__name__}")


# ── pacing ───────────────────────────────────────────────────────────────────
def unanswered(remark_times, owner_turn_at: float) -> int:
    """How many proactive remarks came after the owner's last turn."""
    try:
        t0 = float(owner_turn_at or 0.0)
        return sum(1 for t in (remark_times or ()) if float(t) > t0)
    except Exception:
        return 0


def rate_verdict(now: float, *, remark_times, owner_turn_at: float,
                 last_attempt_at: float, cooldown_s: float, factor: float,
                 max_unanswered: int, attempt_gap_s: float) -> tuple:
    """(ok, reason): may a proactive remark be ATTEMPTED at ``now``?

    One monotonic clock for every argument. ``remark_times``: when each
    spoken remark was made. ``owner_turn_at``: the owner's last turn (0 =
    none). ``last_attempt_at``: the last generation, spoken or dropped.

      * attempt_gap_s after ANY attempt — a dropped remark must not re-ask
        the model every check;
      * n unanswered remarks since the owner last spoke: n >= max_unanswered
        -> quiet until he speaks; else wait cooldown_s * factor**(n-1) after
        the last one — so an ignored remark backs off instead of repeating.
    The reason is numbers only."""
    try:
        now = float(now)
        if last_attempt_at and now - float(last_attempt_at) < attempt_gap_s:
            return (False, f"last attempt {now - float(last_attempt_at):.0f} s "
                           f"ago (gap {attempt_gap_s:.0f} s)")
        n = unanswered(remark_times, owner_turn_at)
        if n <= 0:
            return (True, "")
        if n >= int(max_unanswered):
            return (False, f"{n} unanswered remarks - quiet until the owner "
                           f"speaks")
        wait = float(cooldown_s) * (float(factor) ** (n - 1))
        last = max(float(t) for t in remark_times)
        if now - last < wait:
            return (False, f"{n} unanswered, last {now - last:.0f} s ago "
                           f"(wait {wait:.0f} s)")
        return (True, "")
    except Exception as e:
        return (False, f"rate check failed: {type(e).__name__}")
