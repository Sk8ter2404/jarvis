"""core/teams_unread.py - "any messages I missed?" answered from a real reading.

Live 2026-11-05 14:43: asked by the owner about missed messages, JARVIS said he had no
way to check, and in the same reply gave a count of unread Teams messages that
no code had read. This module holds the pure parts of the teams_unread action
(skills/ms_graph.py): which words ask for it, and the sentence it speaks from
what Microsoft Graph and notification triage actually hold. Stdlib only, no
I/O; never raises.

The reply says counts, names and how long ago, nothing else: the same things
the arrival briefing already speaks. A message body is never read out. When
Graph cannot be read the reply says so and why, rather than guessing.
"""
from __future__ import annotations

import re

TOAST_WINDOW_S = 12 * 3600.0      # "while I was gone": what counts as recent
_TOAST_NAME_MAX_WORDS = 4

# ── which words ask for it ───────────────────────────────────────────────────
# Whole-utterance shapes only: an ordinary sentence that merely contains
# "messages" must reach the model. Another channel named anywhere keeps the
# line away from this action ("any email?", "any text messages?").
_OTHER_CHANNEL_RE = re.compile(
    r"\b(?:e-?mails?|mail|inbox|gmail|outlook|texts?|sms|imessages?|whatsapp|"
    r"discord|slack|signal|telegram|messenger|facebook|instagram|snapchat|"
    r"reddit|linkedin|voicemails?|phone|server|network|router|latency|"
    r"packets?|claude|chatgpt|logs?|errors?|console|terminal|commits?|git)\b")
_NOT_AN_ASK_RE = re.compile(
    r"\b(?:don't|dont|do not|never|stop|send|sending|write|draft|compose|"
    r"delete|clear|mute|forward|reply to|respond|open|close|aren't|weren't|"
    r"without)\b")
_LEAD_NOISE_RE = re.compile(
    r"^(?:hey|ok|okay|yo|so|and|also|um|uh|well|sir|jarvis)\s+")
_POLITE_LEAD = (r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
                r"(?:(?:check|see|tell\s+me|let\s+me\s+know)\s+"
                r"(?:if|whether)\s+)?")
_HAVE = (r"(?:(?:do\s+i\s+have|have\s+i\s+(?:got|had)|did\s+i\s+(?:get|miss|"
         r"have|receive)|i\s+have|i\s+got|i've\s+got|got|are\s+there|"
         r"is\s+there|were\s+there|there\s+are)\s+)?")
_NOUN = r"(?:teams\s+)?(?:messages?|chats?|pings?|dms?|direct\s+messages?)"
_ADJ = r"(?:(?:new|unread|missed|recent)\s+)*"
_ANY_ONE = r"(?:anyone|anybody|someone|somebody)"
# What may follow the ask and change nothing about it.
_TAIL = (r"(?:\s+(?:that\s+)?i\s+(?:have\s+|might\s+have\s+|may\s+have\s+)?"
         r"missed)?"
         r"(?:\s+for\s+me)?"
         r"(?:\s+(?:on|in)\s+teams)?"
         r"(?:\s+(?:while|when|since|after)\s+i\s+(?:was|were|went|left|stepped)"
         r"\b[\w\s']{0,30})?"
         r"(?:\s+(?:today|yet|recently|lately|so\s+far|this\s+(?:morning|"
         r"afternoon|evening)|at\s+all|just\s+now|overnight|please|sir|jarvis))*$")
_ASK_RES = tuple(re.compile("^" + _POLITE_LEAD + body + _TAIL) for body in (
    # "any messages", "do I have any new Teams messages", "did I miss any chats"
    _HAVE + r"any\s+" + _ADJ + _NOUN,
    # "unread Teams messages", "have I got new messages"
    _HAVE + r"(?:new|unread|missed)\s+" + _ADJ + _NOUN,
    # "Teams messages", "messages I missed"
    r"teams\s+(?:messages|chats|pings|dms)",
    r"(?:messages|chats|pings)(?=\s+(?:that\s+)?i\s+(?:\w+\s+){0,2}missed)",
    # "how many unread messages do I have"
    r"how\s+many\s+" + _ADJ + _NOUN + r"(?:\s+(?:do\s+i\s+have|have\s+i\s+got|"
    r"are\s+there|did\s+i\s+(?:get|miss)))?",
    # "did anyone message me", "has anybody pinged me", "anyone message me"
    r"(?:(?:did|has|have)\s+)?" + _ANY_ONE + r"\s+(?:just\s+|been\s+)?"
    r"(?:message|messaged|messaging|ping|pinged|pinging|dm|dmed|dm'd)\s+me",
    r"(?:did|has)\s+" + _ANY_ONE + r"\s+(?:sent|send)\s+me\s+(?:a|any)\s+"
    r"(?:teams\s+)?(?:message|ping|chat)",
    # "who messaged me", "who's been messaging me", "who sent me a message"
    r"who(?:'s|\s+has|\s+have|\s+just)?(?:\s+been)?\s+(?:messaging|pinging|"
    r"messaged|pinged|dmed|dm'd)\s+me",
    r"who(?:'s|\s+has|\s+just)?\s+sent\s+me\s+(?:a|any)\s+"
    r"(?:teams\s+)?(?:message|ping|chat)",
    # "anything new on Teams", "what's new on Teams", "what did I miss on Teams"
    r"(?:anything|something)(?:\s+(?:new|unread))?\s+(?:on|in|from)\s+teams",
    r"(?:what's|whats|what\s+is)\s+new\s+(?:on|in)\s+teams",
    r"what\s+did\s+i\s+miss\s+(?:on|in)\s+teams",
    # "check Teams", "check my Teams messages"
    r"check\s+(?:my\s+)?teams(?:\s+(?:messages|chats))?",
))
_MAX_WORDS = 18


def _normal(text) -> str:
    s = str(text or "").replace("’", "'").replace("‘", "'").lower()
    s = " ".join(re.sub(r"[?!.,;:\"]+", " ", s).split())
    while True:
        s2 = _LEAD_NOISE_RE.sub("", s)
        if s2 == s:
            return s
        s = s2


def asks_teams_messages(text) -> bool:
    """True when the whole line asks whether messages are waiting: "any
    messages I missed", "any Teams messages", "did anyone message me", "who
    messaged me". Another channel, an instruction to send or a line that only
    mentions messages stays False. Never raises."""
    try:
        s = _normal(text)
        if not s or len(s.split()) > _MAX_WORDS:
            return False
        if _OTHER_CHANNEL_RE.search(s) or _NOT_AN_ASK_RE.search(s):
            return False
        return any(rx.match(s) for rx in _ASK_RES)
    except Exception:
        return False


# ── what the toast listener holds ────────────────────────────────────────────
def toast_name(title) -> str:
    """The sender or chat name a Teams toast's title leads with, or "" when
    the title is not a short name. The text after a colon is never kept: that
    is message content."""
    try:
        s = " ".join(str(title or "").split()).split(":")[0].strip(" -|")
        if (not s or "@" in s or re.search(r"\d{5,}", s)
                or len(s.split()) > _TOAST_NAME_MAX_WORDS or len(s) > 40):
            return ""
        return s
    except Exception:
        return ""


def teams_toasts(records, now: float, window_s: float = TOAST_WINDOW_S) -> list:
    """The Teams records among triage's recent toasts, newest first, within
    ``window_s``. Junk records are skipped. Never raises."""
    out = []
    for r in records or ():
        try:
            ts = float(r.get("ts") or 0.0)
            if (re.search(r"(?i)\bteams\b", str(r.get("app") or ""))
                    and str(r.get("action") or "") != "drop"
                    and 0 <= float(now) - ts <= window_s):
                out.append(r)
        except Exception:
            continue
    out.sort(key=lambda r: float(r.get("ts") or 0.0), reverse=True)
    return out


def _ago(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 90:
        return "just now"
    if s < 90 * 60:
        return f"{int(round(s / 60.0))} minutes ago"
    h = int(round(s / 3600.0))
    return f"{h} hours ago"


# ── the sentence ─────────────────────────────────────────────────────────────
def _graph_problem(status) -> str:
    """Why Graph gave nothing, in his words. ``status``: "unconfigured", 0 (no
    token), an HTTP code, or anything else for a failed call."""
    if status in ("unconfigured", 0):
        return ("I can't read Teams directly, sir: Microsoft Graph isn't "
                "signed in here.")
    if status in (401, 403):
        return ("Microsoft Graph turned down the Teams read, sir: this "
                "sign-in isn't allowed to read chats.")
    return "I couldn't reach Microsoft Graph for Teams just now, sir."


def compose(graph, status, toasts, now: float) -> str:
    """The spoken answer. ``graph``: get_teams_unread_count()'s dict, or None
    when Graph gave nothing (``status`` says why). ``toasts``: triage's recent
    records, or None when triage is not loaded. Says only what those hold."""
    recent = None if toasts is None else teams_toasts(toasts, now)
    latest = recent[0] if recent else None
    name = toast_name(latest.get("title")) if latest else ""
    when = _ago(float(now) - float(latest.get("ts") or now)) if latest else ""
    if isinstance(graph, dict):
        try:
            count = int(graph.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        top = str(graph.get("top_sender") or "").strip()
        if count <= 0:
            return "Nothing unread on Teams, sir."
        if count == 1:
            head = "One unread Teams chat, sir" + (f", from {top}." if top else ".")
        elif top:
            head = f"{count} unread Teams chats, sir, the newest from {top}."
        else:
            head = f"{count} unread Teams chats, sir."
        if not latest:
            return head
        same = bool(top) and name.split(" ")[0].lower() == top.lower()
        who = f" from {name}" if name and not same else ""
        return f"{head} The latest Teams notification{who} reached me {when}."
    head = _graph_problem(status)
    if recent is None:
        return head + " So I can't say what's unread."
    if not recent:
        return (head + " No Teams notification has reached me in the last 12 "
                "hours, though that doesn't prove nothing is unread.")
    n = len(recent)
    noun = "One Teams notification" if n == 1 else f"{n} Teams notifications"
    verb = "has" if n == 1 else "have"
    who = f", from {name}" if n == 1 and name else (
        f", the latest from {name}" if name else "")
    return f"{head} {noun} {verb} reached me in the last 12 hours{who}, {when}."
