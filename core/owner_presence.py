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
# is spoken whole, never reduced to a recap fragment.
KEEP_WHOLE_SOURCES = frozenset({
    "morning", "evening", "daily", "news", "recap", "handoff",
    "weekly_digest_briefing", "anticipation_briefing", "focus", "focus_mode",
})

# Lines about a MOMENT (a nudge to take a break, a habit offer, a greeting, a
# banter line): once stale they are simply over, so they are dropped (with a
# log line) rather than recapped.
EXPIRE_QUIETLY_SOURCES = frozenset({
    "wellness", "anticipation", "banter", "greet", "arrival", "posture",
    "night_owl", "character", "screen_watch",
})

_RECAP_NAMED = 3          # fragments named in a recap; the rest are counted
_FRAGMENT_MAX = 70        # characters per recap fragment
_RECAP_SOURCE = "recap:presence"


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


def counts_as_room_talk(text: str) -> bool:
    """A dropped non-wake capture is room talk when it holds at least three
    words - "Thank you." / "You" / "Bye bye." are the classic Whisper noise
    hallucinations on a quiet room, not a conversation (and a conversation
    that is really happening produces longer lines within the hold window)."""
    try:
        words = re.findall(r"[A-Za-z0-9']+", text or "")
        return len(words) >= 3
    except Exception:
        return False


# ── the drain after he comes back ────────────────────────────────────────────

_TAG_RE = re.compile(r"^(?:\s*\[[a-z_]+(?::[^\]]*)?\]\s*)+", re.IGNORECASE)
_LEAD_SIR_RE = re.compile(r"^(?:sir|if i may, sir|pardon me, sir)\s*[,—–-]\s*",
                          re.IGNORECASE)
_TRAIL_SIR_RE = re.compile(r"[,\s]*\bsir\b\s*$", re.IGNORECASE)
_CLAUSE_CUT_RE = re.compile(r"(?<=[^\s])(?:[.!?](?:\s|$)|\s[—–]\s|;\s|\s-\s)")


def recap_fragment(message: str) -> str:
    """The first clause of a queued line, vocative and tags stripped, short
    enough to name in a recap. '' when nothing speakable is left."""
    try:
        text = _TAG_RE.sub("", str(message or "")).strip()
        text = _LEAD_SIR_RE.sub("", text).strip()
        m = _CLAUSE_CUT_RE.search(text)
        if m:
            text = text[:m.start() + 1] if text[m.start()] in ".!?" else text[:m.start()]
        text = text.strip().rstrip(".!?,;:—–- ").strip()
        text = _TRAIL_SIR_RE.sub("", text).strip().rstrip(",;:—–- ").strip()
        if len(text) > _FRAGMENT_MAX:
            cut = text[:_FRAGMENT_MAX].rsplit(" ", 1)[0].rstrip(",;:—–- ")
            text = cut or text[:_FRAGMENT_MAX]
        return text
    except Exception:
        return ""


def recap_line(fragments: list, extra: int = 0) -> str:
    """'While you were away, sir: A; B; and C, plus 2 more.' — '' when there
    is nothing to name. Never asks a question (nothing on the queue may end
    unanswerable in wake-word mode)."""
    frags = [f for f in (fragments or []) if f]
    if not frags:
        return ""
    if len(frags) == 1:
        listed = frags[0]
    else:
        listed = "; ".join(frags[:-1]) + "; and " + frags[-1]
    if extra > 0:
        listed += f", plus {extra} more"
    return f"While you were away, sir: {listed}."


def plan_return_drain(items: list, *, now: float, stale_s: float,
                      exempt=EXEMPT_SOURCES, keep_whole=KEEP_WHOLE_SOURCES,
                      expire_quietly=EXPIRE_QUIETLY_SOURCES) -> tuple:
    """What a drain does with the claimed queue now that the owner is here.

    Returns (items_out, dropped, folded):
      items_out - the entries to speak, in queue order. Every stale STATUS
                  line (non-exempt, not keep-whole, older than stale_s) is
                  replaced by ONE recap entry at the place of the first one;
                  the newest line per source is the one named.
      dropped   - stale moment-lines (expire_quietly) left unspoken.
      folded    - the stale status lines the recap stands for.
    Entries with no usable 'ts' count as fresh. Never raises: on an error the
    queue is returned unchanged."""
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
            if not stale or root in exempt or root in keep_whole:
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
            line = recap_line(named, extra=len(frags) - len(named))
            if line:
                out.insert(recap_at, {"ts": float(now), "message": line,
                                      "source": _RECAP_SOURCE})
        return out, dropped, folded
    except Exception:
        return list(items), [], []
