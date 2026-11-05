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
    # Completed-work claims retired from the phrase pools (2026-10-01).
    "I've run the calculations, sir.",
    "Cross-referencing now.",
    "Cross-referencing now, sir.",
    # An empty promise retired from the phrase pools (v3.0.6).
    "I'll note that for posterity, sir.",
    "I'll note that for posterity.",
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


# ── remarks with nothing behind them (v3.0.11, PROACTIVE_GROUNDED_ONLY) ─────
# Live 2026-11-05: "it's a remarkably productive morning so far" (11:35:27),
# "You've been quite still for some time, sir" (13:07:48), "I couldn't help but
# notice the silence, sir" (14:56:48). JARVIS knows nothing about the owner's
# productivity, and the silence is the very thing that triggered the prompt. A
# fact that merely uses "still" or "noticed" is not caught ("Honey is still
# edible", "Fleming noticed the mould"); a remark is optional, so a fact that
# says "busy" or "productive" may be lost, and that is fine.
_NOTHING_REPLY_RE = re.compile(
    r"^\W*(?:skip|nothing|none|n/?a|no comment)\W*(?:sir\W*)?$")
_UNGROUNDED_RES: tuple = (
    ("the silence", re.compile(r"\bsilen(?:ce|t)\b|\bhush\b")),
    ("the quiet", re.compile(
        r"\b(?:been|so|very|rather|quite|awfully|remarkably|unusually|"
        r"uncharacteristically|it's|it is|all|too)\s+quiet\b|"
        r"\bquiet\s+(?:in here|around here|today|for|lately|so far)\b|"
        r"\bpeace and quiet\b")),
    ("not speaking", re.compile(
        r"\b(?:haven't|have not|hasn't|not)\s+(?:said|spoken|uttered)\b|"
        r"\bsaid a word\b")),
    ("stillness", re.compile(
        r"\bstillness\b|\b(?:haven't|have not|not)\s+moved\b|"
        r"\b(?:quite|very|rather|so|awfully|remarkably|unusually|sitting|sat|"
        r"stayed|remained|been)\s+still\b(?!\s+\w+ing\b)")),
    ("how productive he is", re.compile(
        r"\bproductiv\w+\b|\bbusy\b|\bhard at (?:work|it)\b|"
        r"\bworking (?:\w+ )?hard\b|\bon a roll\b|\bdiligent\w*\b|"
        r"\bgetting (?:a lot|lots|much|plenty) done\b")),
    # JARVIS as the one noticing, not "Fleming noticed the mould".
    ("what JARVIS noticed", re.compile(
        r"\bif i may observe\b|"
        r"\b(?:i|i've|i have|i'd|couldn't help but|can't help but|"
        r"cannot help but|could not help but|hard not to)\b[^.?!]{0,30}?"
        r"\b(?:notic(?:e|ed|ing)|observ(?:e|ed|ing)|sens(?:e|ed|ing)|"
        r"detect(?:ed|ing)?)\b")),
    ("how he seems", re.compile(r"\byou(?:'ve| have) been\b|\byou seem\b")),
)


def ungrounded_remark(text) -> str:
    """What makes ``text`` an empty remark, or "": it says nothing ("SKIP"), or
    talks about the silence, the quiet, stillness, how productive or busy the
    owner is, or what JARVIS has noticed. Never raises."""
    try:
        s = _plain(text).strip()
        if _NOTHING_REPLY_RE.match(s):
            return "said nothing"
        for name, rx in _UNGROUNDED_RES:
            if rx.search(s):
                return f"talks about {name}"
        return ""
    except Exception:
        return ""


def check_remark(text, *, hour: int, recent=(), pool=None,
                 grounded_only: bool = False) -> tuple:
    """(ok, reason) for one generated proactive remark at local ``hour``.

    Dropped (ok False) when it states a time of day the clock contradicts,
    copies a persona example, or repeats one of ``recent``; with
    ``grounded_only`` also when it has nothing behind it (ungrounded_remark:
    the idle remark asks for this, a screen remark built from what is on
    screen does not). The reason names
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
        if grounded_only:
            empty = ungrounded_remark(text)
            if empty:
                return (False, empty)
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
