"""core/owner_presence.py — is the owner here to hear a proactive line?

Pure checks behind bobert_companion._owner_present() and the speech-queue
drain (_speak_pending). Stdlib only, no I/O; the monolith supplies the clocks
and the readings.

THE LIVE INCIDENTS (session_2026-10-01_17-33-02.log, _19-43-10.log,
_21-48-36.log). The owner left the desk at about 18:27 (last physical input
18:27:01, the mic silent until 19:43), and three proactive lines went into the
empty room: the wellness nudge at 19:05:30, the credits login nag at 19:33:46
and the GPU pulse at 19:36:35. At 21:16:10 a wellness nudge talked over people
in the middle of a conversation. The drain (_speak_pending) spoke every queued
source the moment the loop came round; the only gate it consulted was focus
mode. The owner-voice and face gates lived only in should_be_proactive (the
idle remark), and wellness counted any GetLastInputInfo stamp - injected input
included, while an autonomous session was working this PC - as "at the desk".

presence_verdict() is the one presence rule: an owner MIC turn within N
minutes, a SUSTAINED face (core/face_presence.py stamps it), or PHYSICAL
keyboard / mouse input (an injected event never counts). room_talk_recent()
is the "people are talking" hold. plan_return_drain() decides what a drain
does with lines that waited for him: an old status line is not read out on
its own; it is folded into one short recap ("While you were away, sir: ...").
"""
from __future__ import annotations

import math
import re

# Queued sources a drain still speaks while the owner is away or people are
# talking: his OWN reminders (a timer, a scheduled job's line, a promise JARVIS
# made) and the guard-mode intruder alert, whose whole point is to be heard
# when he is NOT at the desk. Compared on the part before any ":" (the same
# rule as bobert_companion._pending_source_in).
EXEMPT_SOURCES = frozenset({"timer", "schedule", "promise", "guard"})

# Lines he asked for that keep their value however late they are: a stale one
# is spoken whole, never reduced to a recap fragment. The printer's lines too
# (2026-10-02 review repair): the fold cut "Slight problem, sir - your H2D
# appears to be unwell. Error code 0300-0100 on layer 142." to "Slight
# problem." - the printer and the code were the whole point.
# "arrival_v2" (2026-11-17): the presence-triggered morning briefing's own
# source was missing, so with PENDING_SOURCE_POLICY off a stale one could be
# cut to a fragment (core/pending_policy.BRIEFING_SOURCES always had it).
KEEP_WHOLE_SOURCES = frozenset({
    "morning", "arrival", "arrival_v2", "evening", "daily", "news", "recap",
    "handoff", "weekly_digest_briefing", "anticipation_briefing", "focus",
    "focus_mode", "bambu", "bambu_voice_companion", "print_companion",
})


def is_urgent(entry) -> bool:
    """A queued entry flagged ``urgent`` (proactive_announce(urgent=True):
    a print failure, a printer error code, a layer shift) is spoken through
    the presence and room-talk holds and is never folded or expired - like
    the EXEMPT_SOURCES, but per line rather than per skill (2026-10-02)."""
    try:
        return isinstance(entry, dict) and entry.get("urgent") is True
    except Exception:
        return False

# Lines about a MOMENT (a nudge to take a break, a habit offer, a greeting, a
# banter line): once stale they are simply over, so they are dropped (with a
# log line) rather than recapped.
EXPIRE_QUIETLY_SOURCES = frozenset({
    "wellness", "anticipation", "banter", "greet", "posture", "night_owl",
    "screen_watch", "screen_remark", "pattern_offer",
})

_RECAP_NAMED = 3          # fragments named in a recap; the rest are counted
_FRAGMENT_MAX = 70        # characters per recap fragment
_RECAP_SOURCE = "recap:presence"


# ── lines he summoned (2026-11-17) ───────────────────────────────────────────
# A queued line may carry ``summoned`` (proactive_announce(summoned=...)): he
# caused it a moment ago, so for a short window it passes the holds named
# here - never "owner away" or "gaming", and never a speech hold. Per kind:
#   (holds it may pass, needs his face or his voice - polled input is not
#    enough, needs a quiet room - no music or media now or a moment ago)
#   clap    - "You rang, sir?" answers his double clap. Live 11-16/11-17 11 of
#             15 claps got no answer: the ack waited behind the 45 s room-talk
#             hold and expired at its 10 s age. Many of those claps came with
#             music, so the room must be quiet too (re-checked at speak time).
#             It also passes the short gap after a drain-budget deferral
#             (PENDING_DEFER_GAP_S), which would outlast its 10 s life.
#   arrival - the presence-triggered morning briefing. Live 11-17 06:35:30 it
#             was queued as his face appeared and held behind his video and
#             room talk while he was getting ready, then "owner away" for
#             hours, and dropped unheard at 16:02:41.
# The hold kinds are bobert_companion._presence_hold_reason's (and the drain's
# deferral gap) before any " (".
SUMMONED_RULES = {
    "clap":    (frozenset({"room talk", "after the drain budget"}), True, True),
    "arrival": (frozenset({"room talk", "media playing"}), False, False),
}


def summoned_kind(entry) -> str:
    """The entry's ``summoned`` kind when it is one SUMMONED_RULES knows."""
    try:
        kind = entry.get("summoned") if isinstance(entry, dict) else None
        return kind if isinstance(kind, str) and kind in SUMMONED_RULES else ""
    except Exception:
        return ""


def summoned_passes(entry, *, hold_kind: str, age_s, window_s,
                    present, seen_or_heard, room_quiet) -> bool:
    """May this summoned line be spoken through ``hold_kind`` now? Only a
    hold its kind lists, only within ``window_s`` of being queued (0 = off),
    only while ``present()`` (presence_verdict); a kind that needs it also
    ``seen_or_heard()`` (a face or his voice) and ``room_quiet()``. The three
    readings are callables, read only when needed. Never raises (False)."""
    try:
        rule = SUMMONED_RULES.get(summoned_kind(entry))
        if not rule or hold_kind not in rule[0]:
            return False
        window = float(window_s or 0)
        age = float(age_s)
        if window <= 0 or math.isnan(age) or age > window:
            return False
        if not present():
            return False
        if rule[1] and not seen_or_heard():
            return False
        if rule[2] and not room_quiet():
            return False
        return True
    except Exception:
        return False


def source_root(entry) -> str:
    """The queued entry's source before any ':' ('' when it has none)."""
    try:
        src = entry.get("source") if isinstance(entry, dict) else None
        return src.split(":", 1)[0] if isinstance(src, str) else ""
    except Exception:
        return ""


def _fresh(age_s, window_s) -> bool:
    try:
        age = float(age_s)
        return (not math.isnan(age)) and 0.0 <= age <= float(window_s)
    except (TypeError, ValueError):
        return False


def presence_verdict(*, voice_age_s, face_age_s, input_age_s,
                     voice_window_s: float, face_window_s: float,
                     input_window_s: float) -> tuple:
    """(present, why). Each age is seconds since that signal (None or inf =
    never / unknown). Present when ANY signal is inside its window:
      voice  - an accepted owner MIC turn (typed / injected turns never count)
      face   - a sustained face a camera confirmed
      input  - physical keyboard / mouse input (injected input never counts)
    `why` names the signal that answered, or every stale one when away."""
    if _fresh(voice_age_s, voice_window_s):
        return True, f"owner spoke {float(voice_age_s):.0f} s ago"
    if _fresh(face_age_s, face_window_s):
        return True, f"face seen {float(face_age_s):.0f} s ago"
    if _fresh(input_age_s, input_window_s):
        return True, f"physical input {float(input_age_s):.0f} s ago"

    def _ago(age, what):
        try:
            a = float(age)
        except (TypeError, ValueError):
            return f"no {what}"
        if math.isnan(a) or math.isinf(a) or a < 0:
            return f"no {what}"
        return f"{what} {a / 60.0:.0f} min ago"
    return False, ", ".join((_ago(voice_age_s, "owner voice"),
                             _ago(face_age_s, "face"),
                             _ago(input_age_s, "physical input")))


def room_talk_recent(talk_age_s, window_s: float) -> bool:
    """True when non-wake speech (people, or a show, talking in the room) was
    captured within window_s seconds."""
    return _fresh(talk_age_s, window_s)


# Whisper's stock lines - the outro of every video it was trained on - that
# it writes over music, a reel or room noise (live 22:00-22:24 on 2026-10-01:
# "I'll see you next time." 7 times, "I'm going to show you what I'm going to
# show you." about 7 times). The whole line is the phrase, nothing more.
_STOCK_LINE_RE = re.compile(
    r"^(?:(?:and\s+)?(?:i'?ll\s+|we'?ll\s+)?see\s+you\s+(?:guys\s+)?"
    r"(?:next\s+time|in\s+the\s+next\s+(?:one|video|episode)|soon|later)|"
    r"(?:thanks|thank\s+you)\s+(?:so\s+much\s+)?for\s+watching"
    r"(?:\s+and\s+(?:please\s+)?(?:like\s+and\s+)?subscribe)?|"
    r"(?:please\s+)?(?:like\s+and\s+)?subscribe(?:\s+to\s+(?:my|the|our)"
    r"\s+channel)?|bye(?:\s+bye)?(?:\s+everyone)?)$")


def _self_repeating(words: list) -> bool:
    """One phrase said twice ("I'm going to show you what I'm going to show
    you"): the line's first run of three or more words comes back later."""
    n = len(words)
    for size in range(min(6, n // 2), 2, -1):
        head = words[:size]
        for i in range(size, n - size + 1):
            if words[i:i + size] == head:
                return True
    return False


def counts_as_room_talk(text: str) -> bool:
    """A dropped non-wake capture is room talk when it holds at least three
    words - "Thank you." / "You" / "Bye bye." are the classic Whisper noise
    hallucinations on a quiet room, not a conversation (and a conversation
    that is really happening produces longer lines within the hold window).
    Whisper's stock outro lines and a line that repeats its own opening are
    noise too (2026-10-02)."""
    try:
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text or "")]
        if len(words) < 3:
            return False
        if _STOCK_LINE_RE.match(" ".join(words).replace(",", "")):
            return False
        return not _self_repeating(words)
    except Exception:
        return False


# ── the drain after he comes back ────────────────────────────────────────────

_TAG_RE = re.compile(r"^(?:\s*\[[a-z_]+(?::[^\]]*)?\]\s*)+", re.IGNORECASE)
_LEAD_SIR_RE = re.compile(r"^(?:sir|if i may, sir|pardon me, sir)\s*[,—–-]\s*",
                          re.IGNORECASE)
_TRAIL_SIR_RE = re.compile(r"[,\s]*\bsir\b\s*$", re.IGNORECASE)
_CLAUSE_CUT_RE = re.compile(r"(?<=[^\s])(?:[.!?](?:\s|$)|\s[—–]\s|;\s|\s-\s)")


_MID_SIR_RE = re.compile(r",\s*sir\b(?=\s*[,—–;:.!?-])", re.IGNORECASE)
_FRAGMENT_MIN_WORDS = 4     # a shorter first clause takes the next one too


def recap_fragment(message: str) -> str:
    """The opening of a queued line - its first clause, or as many clauses
    as it takes to say at least _FRAGMENT_MIN_WORDS words ("Slight problem"
    alone says nothing, 2026-10-02) - with the vocative and tags stripped,
    short enough to name in a recap. '' when nothing speakable is left."""
    try:
        text = _TAG_RE.sub("", str(message or "")).strip()
        text = _LEAD_SIR_RE.sub("", text).strip()
        text = _MID_SIR_RE.sub("", text)
        start = 0
        while True:
            m = _CLAUSE_CUT_RE.search(text, start)
            if not m:
                break
            cut = m.start() + 1 if text[m.start()] in ".!?" else m.start()
            if len(text[:cut].split()) >= _FRAGMENT_MIN_WORDS:
                text = text[:cut]
                break
            start = m.end()
        text = text.strip().rstrip(".!?,;:—–- ").strip()
        text = _TRAIL_SIR_RE.sub("", text).strip().rstrip(",;:—–- ").strip()
        if len(text) > _FRAGMENT_MAX:
            cut = text[:_FRAGMENT_MAX].rsplit(" ", 1)[0].rstrip(",;:—–- ")
            text = cut or text[:_FRAGMENT_MAX]
        return text
    except Exception:
        return ""


_COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten")


def _count_word(n: int) -> str:
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else str(n)


def _listed(frags: list) -> str:
    if len(frags) == 1:
        return frags[0]
    return "; ".join(frags[:-1]) + "; and " + frags[-1]


def recap_line(fragments: list, extra: int = 0, *, away: bool = True) -> str:
    """'While you were away, sir: A; B; and C.' — '' when there is nothing
    to name. ``away`` False (the lines waited for another reason: room talk,
    standby, a speech hold - he never left) says 'Earlier, sir:' instead
    (2026-10-02). With ``extra`` more not named it ends 'And two more -
    shall I list them?' (2026-11-17; it was ', plus 2 more.', which nothing
    could ask about): plan_return_drain records the others in the entry's
    history_text, so "JARVIS, yes" reaches the brain with them - the one
    question a queued recap may ask."""
    frags = [f for f in (fragments or []) if f]
    if not frags:
        return ""
    lead = "While you were away, sir" if away else "Earlier, sir"
    line = f"{lead}: {_listed(frags)}."
    if extra > 0:
        line += f" And {_count_word(int(extra))} more - shall I list them?"
    return line


def recap_history(fragments: list, rest: list, *, away: bool = True) -> str:
    """What conversation_history records for a recap whose ``rest`` was not
    named (proactive_announce's history_text rule): the spoken line plus
    the others, so his "yes" has them. '' when nothing is left out."""
    rest = [f for f in (rest or []) if f]
    line = recap_line(fragments, extra=len(rest), away=away)
    if not line or not rest:
        return ""
    return (f"{line} (The other {_count_word(len(rest))}: "
            f"{'; '.join(rest)}.)")


def plan_return_drain(items: list, *, now: float, stale_s: float,
                      exempt=EXEMPT_SOURCES, keep_whole=KEEP_WHOLE_SOURCES,
                      expire_quietly=EXPIRE_QUIETLY_SOURCES,
                      away: bool = True, never_fold=None) -> tuple:
    """What a drain does with the claimed queue now that the owner is here.

    Returns (items_out, dropped, folded):
      items_out - the entries to speak, in queue order. Every stale STATUS
                  line (non-exempt, not keep-whole, older than stale_s) is
                  replaced by ONE recap entry at the place of the first one;
                  the newest line per source is the one named.
      dropped   - stale moment-lines (expire_quietly) left unspoken.
      folded    - the stale status lines the recap stands for.
    Entries with no usable 'ts' count as fresh; an urgent entry (is_urgent)
    is never folded or dropped. ``away``: whether the owner was away while
    they waited - only then does the recap say "While you were away"
    (recap_line). ``never_fold`` (v3.0.11, core/pending_policy.never_fold): a
    predicate for lines that stay whole however stale they are. Never raises:
    on an error the queue is returned unchanged."""
    try:
        out: list = []
        dropped: list = []
        folded: list = []
        recap_at = -1
        for item in items:
            if not isinstance(item, dict):
                out.append(item)
                continue
            root = source_root(item)
            try:
                age = float(now) - float(item.get("ts"))
            except (TypeError, ValueError):
                age = 0.0
            stale = age > float(stale_s)
            if (not stale or root in exempt or root in keep_whole
                    or is_urgent(item)
                    or (never_fold is not None and never_fold(item))):
                out.append(item)
                continue
            if root in expire_quietly:
                dropped.append(item)
                continue
            if recap_at < 0:
                recap_at = len(out)
            folded.append(item)
        if folded:
            newest: dict = {}
            order: list = []
            for item in folded:
                key = source_root(item) or str(item.get("message", ""))[:40]
                if key not in newest:
                    order.append(key)
                newest[key] = item
            frags = []
            for key in order:
                frag = recap_fragment(newest[key].get("message", ""))
                if frag and frag not in frags:
                    frags.append(frag)
            named = frags[:_RECAP_NAMED]
            rest = frags[_RECAP_NAMED:]
            line = recap_line(named, extra=len(rest), away=away)
            if line:
                entry = {"ts": float(now), "message": line,
                         "source": _RECAP_SOURCE}
                if rest:
                    entry["history_text"] = recap_history(named, rest,
                                                          away=away)
                out.insert(recap_at, entry)
        return out, dropped, folded
    except Exception:
        return list(items), [], []


# ── "Did anything happen while I was away?" (2026-11-17) ────────────────────
# Live 16:02:13: "did you log anything while I was at school today?" went to
# the brain, which said "I've checked the logs, sir - nothing worth noting"
# with no action (the validator blocked it), then ran agents_status (Claude
# chats). The real away recap only came 28 s later, from the queue drain
# (16:02:41 "While you were away, sir: ..."). bobert_companion.
# _away_recap_shortcut answers these questions from the queue, before the
# model is called. Both halves are needed: an away clause ("while I was at
# school / away / gone / out") AND a generic ask ("did you log anything", "did
# anything happen", "what happened", "anything come up"). A question naming
# its own subject (Teams, email, a print, the server, ...) is left to the
# router and the brain: it has its own action.
_AWAY_CLAUSE_RE = re.compile(
    r"\b(?:while|when)\s+i\s*(?:was|were|'?ve\s+been|have\s+been)\s+"
    r"(?:away|gone|out|asleep|sleeping|at\s+(?:school|work|class|the\s+gym|"
    r"uni|college|the\s+store))\b"
    r"|\bsince\s+i\s+(?:left|got\s+back|was\s+gone)\b", re.IGNORECASE)
_AWAY_ASK_RE = re.compile(
    r"\b(?:did|have)\s+you\s+(?:log|logged|note|noted|record|recorded|catch|"
    r"caught|notice|noticed|see|seen)\s+anything\b"
    r"|\b(?:did|has)\s+anything\s+(?:happen|happened|come\s+up|change|"
    r"changed|go\s+on)\b"
    r"|\banything\s+(?:happen|happened|come\s+up|new|i\s+should\s+know|"
    r"to\s+report|worth\s+knowing)\b"
    r"|\bwhat\s+(?:happened|went\s+on|did\s+i\s+miss|came\s+up)\b",
    re.IGNORECASE)
_AWAY_OWN_SUBJECT_RE = re.compile(
    r"\b(?:teams|e-?mails?|mail|inbox|messages?|texts?|notifications?|calls?|"
    r"voicemail|discord|slack|tv|show|episode|print|prints|printer|printing|"
    r"server|unraid|plex|backup|download|downloads|stock|price|game|score|"
    r"weather|news|package|delivery|door|camera|cameras)\b", re.IGNORECASE)

AWAY_NOTHING_WAITING_LINE = (
    "Nothing was waiting for you from while you were away, sir.")
AWAY_STILL_HELD_LINE = (
    "A few notices are still waiting, sir - I'll read them out in a moment.")
AWAY_RECAP_REPEAT_S = 3 * 3600.0   # a recap said this recently is said again
# Lines dropped unsaid while he was away (core/queue_receipts.missed_phrase,
# 2026-11-17): named after the answer.
# Worded so it stays true when a line went stale while he was here too.
AWAY_MISSED_LINE = ("Also, {missed} went unsaid, sir - out of date before "
                    "I could say it.")
AWAY_ONLY_MISSED_LINE = ("Nothing is waiting now, sir, but {missed} went "
                         "unsaid - out of date before I could say it.")


def is_away_recap_ask(text) -> bool:
    """True for a generic "did anything happen while I was away" question:
    an away clause plus a generic ask, and no subject of its own (see
    _AWAY_OWN_SUBJECT_RE). Never raises."""
    try:
        t = " ".join(str(text or "").replace("\u2019", "'").split())
        if not t:
            return False
        return bool(_AWAY_CLAUSE_RE.search(t) and _AWAY_ASK_RE.search(t)
                    and not _AWAY_OWN_SUBJECT_RE.search(t))
    except Exception:
        return False


def away_answer_line(*, spoke: bool, still_queued: bool = False,
                     last_recap: str = "", last_recap_age_s=None,
                     missed: str = "") -> str:
    """What JARVIS says after draining the queue for an away question.
    '' when the drain spoke (the lines themselves were the answer). Else the
    recap he already gave within AWAY_RECAP_REPEAT_S, word for word; else
    "still waiting" when lines are held on the queue (a speech hold); else
    the honest "nothing was waiting". ``missed`` (2026-11-17, core/
    queue_receipts.missed_phrase: "the morning briefing and two memory
    warnings") names what was dropped unsaid: AWAY_MISSED_LINE after the
    answer (alone when the drain spoke), or AWAY_ONLY_MISSED_LINE instead of
    "nothing was waiting". Never raises."""
    try:
        missed = str(missed or "").strip()
        extra = AWAY_MISSED_LINE.format(missed=missed) if missed else ""
        if spoke:
            return extra
        recap = str(last_recap or "").strip()
        try:
            age = float(last_recap_age_s)
        except (TypeError, ValueError):
            age = None
        if recap and age is not None and 0 <= age <= AWAY_RECAP_REPEAT_S:
            return f"{recap} {extra}".strip()
        if still_queued:
            return f"{AWAY_STILL_HELD_LINE} {extra}".strip()
        if missed:
            return AWAY_ONLY_MISSED_LINE.format(missed=missed)
        return AWAY_NOTHING_WAITING_LINE
    except Exception:
        return AWAY_NOTHING_WAITING_LINE
