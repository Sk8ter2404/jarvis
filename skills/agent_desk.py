"""Agent desk: "what are my agents doing / is anything waiting on me?"
(v3 W19, 2026-10-27, owner reels 7 + 8).

JARVIS only READS the Agent Office (the owner's pixel-art office that watches
every Claude chat and sub-agent; D:/Nextcloud/Pictures/Claude Code/Agent-Office,
spec in its SPEC.md, "State JSON"): one GET of AGENT_OFFICE_URL + /api/state,
2 s timeout, nothing written, nothing started. No Claude Code hook is installed
and ~/.claude/settings.json is never touched - the Office already watches the
transcripts, so JARVIS asks it.

  * agents_status (SPEAK_VERBATIM) - at most three short sentences: how many
    chats are working or waiting, who is waiting on the owner and since when,
    and any alarms. "anything waiting on me" (arg "waiting") narrows it to the
    needs_braden list. Any other arg is a chat's NAME (3.0.1, live 2026-10-27
    "is Ultron online?" got "I can't confirm that"): that one chat's status,
    how long and its activity; no match says so, then the overall summary.
    The Office not running answers "The Agent Office isn't running, sir."
  * Optional proactive line (AGENT_DESK_ANNOUNCE, default off): a NEW
    needs_braden item is announced once through proactive_announce - debounced
    (it must still be there on the next poll, one line per GAP_S, at most four
    an hour), held in the PHONE_PING quiet hours, and a restart never
    re-announces what was already waiting.

Everything that comes out of the Office (session names, activity, needs_braden
text, alarm text) is DATA from outside JARVIS. It is scrubbed (credential
shapes), stripped of emoji and of every bracket / "ACTION" token, clipped, and
only ever SPOKEN: agents_status is a SPEAK_VERBATIM action, so describe()'s
finished sentences are voiced word for word and never go to the follow-up
brain, whose reply parse_and_run_actions would run (security review NET-2,
2026-10-27; it used to be INFORMATIVE, which did hand the text to that round).
It never starts an action. No hook design from reel 7's agent-floor is ported, so no credit line
is owed here.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import threading
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8795"
STATE_PATH = "/api/state"
TIMEOUT_S = 2.0
MAX_BYTES = 1_000_000
FIRST_POLL_DELAY_S = 45.0
GAP_S = 300.0            # minimum seconds between two proactive lines
MAX_PER_HOUR = 4
SEEN_KEEP = 200
NOT_RUNNING = "The Agent Office isn't running, sir."
UNREADABLE = "The Agent Office answered with something I couldn't read, sir."

# A plain literal at module level: tests/test_audit_action_reachability.py
# reads it with ast (it never imports the skill); cleared below when off.
PROMPT_EXAMPLES = (
    "AGENT DESK (skills/agent_desk.py - READ-ONLY, reads the Agent Office):\n"
    "  agents_status - what the owner's Claude chats and agents are doing: how\n"
    "                  many are working or waiting, who is waiting on him and\n"
    "                  since when, and any alarms. Add 'waiting' to narrow it\n"
    # No apostrophe inside a quoted trigger: core.prompt_router pairs the
    # single quotes to find them.
    "                  to what needs him, or the NAME of a chat to answer about that\n"
    "                  one chat (working, idle, waiting on him or offline, for\n"
    "                  how long, and its activity). The chats are his own\n"
    "                  (Ultron is one); never say you cannot check them. Fire\n"
    "                  on 'what are my agents doing', 'how are my agents',\n"
    "                  'status of my chats', 'which chats are working',\n"
    "                  'anything waiting on me', 'is any chat waiting on me',\n"
    "                  'do any agents need me', 'is the agent office up',\n"
    "                  'is Ultron online', 'is the Ops hub working'.\n"
    "  'what are my agents doing' -> [ACTION: agents_status]\n"
    "  'anything waiting on me' -> [ACTION: agents_status, waiting]\n"
    "  'is Ultron online?' -> [ACTION: agents_status, Ultron]\n"
    "  'what is Ultron doing?' -> [ACTION: agents_status, Ultron]\n"
    "  'is the Ops hub working?' -> [ACTION: agents_status, Ops hub]\n"
)

_lock = threading.Lock()
_stop = threading.Event()
_seen: dict = {}          # key -> ts: needs_braden items already known / said
_pending: set = set()     # keys seen once, announced only if seen again
_baselined = [False]
_sent_at: list = []       # timestamps of proactive lines


# ── settings ───────────────────────────────────────────────────────────
def _cfg(name, default):
    try:
        from core import config as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default


def enabled() -> bool:
    return bool(_cfg("AGENT_DESK_ENABLED", True))


def announce_on() -> bool:
    return bool(_cfg("AGENT_DESK_ANNOUNCE", False))


# Declared only while the feature is on (read once, at import): with
# AGENT_DESK_ENABLED off the skill adds nothing to the system prompt and
# nothing to the global speak-verbatim set, so the brain is never told
# about an action that is not registered. The skill loader ignores "" / ().
# (A plain literal first, then cleared: tests read the literal from source.)
# VERBATIM, never INFORMATIVE: Office text is outside data and must not reach
# the action-running follow-up round (NET-2).
SPEAK_VERBATIM_ACTIONS = ("agents_status",)
if not enabled():
    SPEAK_VERBATIM_ACTIONS = ()
    PROMPT_EXAMPLES = ""


def office_url() -> str:
    u = str(_cfg("AGENT_OFFICE_URL", DEFAULT_URL) or "").strip().rstrip("/")
    return u if re.match(r"^https?://[A-Za-z0-9._:\[\]-]+$", u) else DEFAULT_URL


# ── reading the Office ─────────────────────────────────────────────────
def _fetch(url: str, timeout: float = TIMEOUT_S) -> dict:
    """GET and decode the state JSON. Raises on any fault."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener.open(req, timeout=timeout) as resp:
        raw = resp.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("state too large")
    data = json.loads(raw.decode("utf-8", "replace"))
    if not isinstance(data, dict):
        raise ValueError("state is not an object")
    return data


def read_state(fetch=None):
    """``(state, error)``: error is "down" (nothing answered) or "bad" (an
    answer that is not the state JSON); never raises."""
    fetch = fetch or _fetch
    try:
        return fetch(office_url() + STATE_PATH, TIMEOUT_S), ""
    except ValueError:                  # includes json.JSONDecodeError
        return None, "bad"
    except Exception:
        return None, "down"


# ── making outside text safe to say ────────────────────────────────────
_ACTION_RE = re.compile(r"(?i)\baction\b\s*:?")
_KEEP_RE = re.compile(r"[^A-Za-z0-9 .,'&/+_#:%$-]")

# core.failure_markers substrings make the monolith treat an action RESULT as a
# failed action (failure follow-up, re-prompt, the failure branch of a chain).
# A chat called "Fix failed build" must not turn a correct answer into one, so
# marker words in outside text are reworded before they are spoken.
_DEFANG = (
    (r"could\s+not", "cannot"), (r"couldn['’]t", "cannot"),
    (r"can['’]t", "cannot"), (r"didn['’]t", "did not"),
    (r"wouldn['’]t", "would not"),
    (r"failed", "fail"), (r"refused", "declined"),
    (r"no\s+tracks\s+found", "no tracks"),
    (r"no\s+window\s+matching", "no window"),
    (r"unknown(?=\s)", "unrecognised"), (r"format:", "format -"),
)
_DEFANG_RES = tuple((re.compile(p, re.I), r) for p, r in _DEFANG)


def defang(text: str) -> str:
    """Reword every core.failure_markers substring out of ``text`` (case-
    insensitive, idempotent). Whatever the table misses is dropped outright,
    so no marker survives even if the marker list grows."""
    s = str(text)
    for rx, rep in _DEFANG_RES:
        s = rx.sub(rep, s)
    try:
        from core.failure_markers import FAILURE_MARKERS
        for m in FAILURE_MARKERS:
            s = re.sub(re.escape(m), " ", s, flags=re.I)
    except Exception:
        pass
    return s


def _scrub(text: str) -> str:
    try:
        from core.phone_ping import scrub
        out, withheld = scrub(text)
        return "details withheld" if withheld else out
    except Exception:
        return re.sub(r"(?i)\b(password|passwd|token|api[ _-]?key|secret)\b\S*\s*\S*",
                      "[redacted]", text)


def clean(text, limit: int = 40, head_only: bool = False) -> str:
    """One piece of Office text, safe to speak: credential shapes redacted,
    emoji / brackets / braces / angle marks / the word ACTION removed, cut at
    a dash separator when ``head_only``, clipped to ``limit`` characters."""
    s = str(text if text is not None else "")
    s = _scrub(s)
    if head_only:
        s = re.split(r"\s[-\u2013\u2014]+\s", s, maxsplit=1)[0]
    s = _ACTION_RE.sub(" ", s)
    s = _KEEP_RE.sub(" ", s)
    s = defang(s)
    s = re.sub(r"\s+", " ", s).strip(" -_:.,")
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].rstrip(" -_:.,") or s[:limit]
    return s


def _num(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _clock(ts: float) -> str:
    return time.strftime("%I:%M %p", time.localtime(ts)).lstrip("0")


def since_phrase(ts, now=None) -> str:
    """"for 12 minutes" / "for about 3 hours" / "since 9:05 PM"; "" if unknown."""
    t = _num(ts)
    if t is None:
        return ""
    now = time.time() if now is None else now
    age = max(0.0, now - t)
    if age < 90:
        return "just now"
    if age < 3600:
        m = int(round(age / 60))
        return f"for {m} minute{'' if m == 1 else 's'}"
    if age < 6 * 3600:
        h = int(round(age / 3600))
        return f"for about {h} hour{'' if h == 1 else 's'}"
    lt, nt = time.localtime(t), time.localtime(now)
    if lt.tm_yday == nt.tm_yday and lt.tm_year == nt.tm_year:
        return f"since {_clock(t)}"
    return f"since {time.strftime('%A', lt)} {_clock(t)}"


# ── digesting the state ────────────────────────────────────────────────
_WAIT_STATES = ("waiting_braden", "waiting_permission")
_ALARM_WORDS = {
    "stuck": ("a stuck chat", "stuck chats"),
    "clash": ("a file clash", "file clashes"),
    "gpu_contention": ("GPU contention", "GPU contention alarms"),
    "error": ("an error", "errors"),
    "runaway": ("a runaway agent", "runaway agents"),
    "needs_braden": ("something waiting on you over an hour",
                     "things waiting on you over an hour"),
}


def _items(value) -> list:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def waiting_list(state: dict) -> list:
    """The things waiting on the owner as ``[{"who", "what", "since"}]``,
    oldest first: the needs_braden board plus any chat whose own status says
    it is waiting (de-duplicated by name)."""
    out, seen = [], set()
    for it in _items(state.get("needs_braden")):
        who = clean(it.get("source"), 40, head_only=True) or "a chat"
        seen.add(who.lower())
        out.append({"who": who, "what": clean(it.get("what"), 70),
                    "since": _num(it.get("since"))})
    claude = state.get("claude") if isinstance(state.get("claude"), dict) else {}
    for s in _items(claude.get("sessions")):
        if s.get("status") in _WAIT_STATES:
            who = clean(s.get("name"), 40, head_only=True) or "a chat"
            if who.lower() in seen:
                continue
            seen.add(who.lower())
            out.append({"who": who, "what": clean(s.get("activity"), 70),
                        "since": _num(s.get("since"))})
    out.sort(key=lambda w: (w["since"] is None, w["since"] or 0.0))
    return out


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _waiting_sentence(waits: list, now) -> str:
    if not waits:
        return ""
    if len(waits) == 1:
        w = waits[0]
        line = f"{w['who']} is waiting on you"
        sp = since_phrase(w["since"], now)
        if sp:
            line += f" {sp}"
        if w["what"]:
            line += f": {w['what']}"
        return line + "."
    bits = []
    for w in waits[:3]:
        bits.append(f"{w['who']} {since_phrase(w['since'], now)}".strip())
    more = len(waits) - len(bits)
    tail = f" and {more} more" if more > 0 else ""
    return f"{len(waits)} chats are waiting on you: " + ", ".join(bits) + tail + "."


def _alarm_sentence(state: dict) -> str:
    counts: dict = {}
    first_what: dict = {}
    for a in _items(state.get("alarms")):
        k = str(a.get("kind") or "").strip().lower()
        counts[k] = counts.get(k, 0) + 1
        if k not in first_what:
            first_what[k] = clean(a.get("what"), 60)
    if not counts:
        return ""
    bits = []
    for k, c in counts.items():
        if k in _ALARM_WORDS:
            one, many = _ALARM_WORDS[k]
            bits.append(one if c == 1 else f"{c} {many}")
        elif c == 1 and first_what.get(k):
            # a kind the Office added later: say its own (cleaned) description
            bits.append(first_what[k])
        else:
            bits.append("an unrecognised alarm" if c == 1 else f"{c} unrecognised alarms")
    total = sum(counts.values())
    head = "One alarm" if total == 1 else f"{total} alarms"
    return head + ": " + ", ".join(bits) + "."


def describe(state: dict, focus_waiting: bool = False, now=None) -> str:
    """The spoken summary: at most three sentences. Never raises."""
    try:
        now = time.time() if now is None else now
        claude = state.get("claude") if isinstance(state.get("claude"), dict) else {}
        sessions = [s for s in _items(claude.get("sessions"))
                    if s.get("status") != "offline"]
        waits = waiting_list(state)
        wsent = _waiting_sentence(waits, now)
        alarms = _alarm_sentence(state)
        if focus_waiting:
            return defang(" ".join(x for x in (
                wsent or "Nothing is waiting on you, sir.", alarms) if x))
        if not sessions:
            first = "I don't see any Claude chats in the office, sir."
        else:
            working = sum(1 for s in sessions if s.get("status") == "working")
            subs = sum(1 for s in sessions for a in _items(s.get("agents"))
                       if a.get("status") == "working")
            first = f"Of {_n(len(sessions), 'chat', 'chats')}, "
            first += (f"{working} {'is' if working == 1 else 'are'} working"
                      if working else "none are working")
            if waits:
                first += f" and {len(waits)} {'is' if len(waits) == 1 else 'are'} waiting on you"
            if subs:
                first += f", with {_n(subs, 'sub-agent', 'sub-agents')} running"
            first += ", sir."
        # defang again: words from two pieces can meet across a join
        return defang(" ".join(x for x in (first, wsent, alarms) if x))
    except Exception:
        return UNREADABLE


_FOCUS_RE = re.compile(r"(?i)\b(wait|waiting|need|needs|me|mine|attention)\b")


_STATUS_WORDS = {
    "working": "working",
    "idle": "idle",
    "waiting_braden": "waiting on you",
    "waiting_permission": "waiting on you",
    "error": "in an error state",
}
# Words that ask about the whole office, not one chat.
_GENERIC_NAMES = frozenset((
    "agent", "agents", "chat", "chats", "all", "everything", "everyone",
    "status", "office", "the office", "agent office", "session", "sessions"))


def _query_name(arg) -> str:
    q = clean(arg, 40)
    q = re.sub(r"(?i)^(the|my|all|all my|all the)\s+", "", q)
    q = re.sub(r"(?i)\s+(chats?|sessions?|agents?)$", "", q)
    # Nothing left but punctuation ("!!") would match every chat as a part.
    if q.lower() in _GENERIC_NAMES or len(_norm(q)) < 2:
        return ""
    return q


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _find_chats(state: dict, query: str) -> list:
    """The chats whose cleaned name or its head before the dash is ``query``
    (case-insensitive); if none is an exact match, those that contain it as
    whole words. ``[(display name, session)]``."""
    query = _norm(query)
    exact, partial = [], []
    rx = re.compile(r"(?<![a-z0-9])" + re.escape(query) + r"(?![a-z0-9])")
    claude = state.get("claude") if isinstance(state.get("claude"), dict) else {}
    for s in _items(claude.get("sessions")):
        head = clean(s.get("name"), 40, head_only=True)
        full = clean(s.get("name"), 80)
        hl, fl = _norm(head), _norm(full)
        shown = head or full or "a chat"
        if query in (hl, fl):
            exact.append((shown, s))
        elif rx.search(hl) or rx.search(fl):
            partial.append((shown, s))
    return (exact or partial)


def _chat_phrase(session: dict, now) -> str:
    """"online and working for 12 minutes" / "offline" for one chat."""
    status = str(session.get("status") or "").strip().lower()
    since = since_phrase(session.get("since"), now)
    if status == "offline":
        return ("offline " + since).strip()
    word = _STATUS_WORDS.get(status) or clean(status.replace("_", " "), 30)
    return ("online and " + word + (" " + since if since else "")
            if word else "online" + (" " + since if since else ""))


def describe_chat(state: dict, query: str, now=None) -> str:
    """One or two sentences about the chat called ``query``; the overall
    summary after a sentence saying so when none matches. Never raises."""
    try:
        now = time.time() if now is None else now
        hits = _find_chats(state, query)
        if not hits:
            return defang(f"I don't see a chat called {clean(query, 40)} in the "
                          f"office, sir. " + describe(state, now=now))
        if len(hits) == 1:
            name, s = hits[0]
            line = f"{name} is {_chat_phrase(s, now)}, sir."
            act = clean(s.get("activity"), 70)
            if act and str(s.get("status") or "").strip().lower() != "offline":
                line += f" Activity: {act}."
            return defang(line)
        bits = [f"{n} is {_chat_phrase(s, now)}" for n, s in hits[:3]]
        more = len(hits) - len(bits)
        tail = f", and {more} more" if more > 0 else ""
        return defang(f"{len(hits)} chats match {clean(query, 40)}: "
                      + "; ".join(bits) + tail + ".")
    except Exception:
        return UNREADABLE


def agents_status(arg: str = "") -> str:
    state, err = read_state()
    if err == "down":
        return NOT_RUNNING
    if err or state is None:
        return UNREADABLE
    if not _FOCUS_RE.search(str(arg or "")):
        query = _query_name(arg)
        if query:
            return describe_chat(state, query)
    return describe(state, focus_waiting=bool(_FOCUS_RE.search(str(arg or ""))))


# ── proactive line for NEW needs_braden items ──────────────────────────
def _minutes(hhmm, default: int) -> int:
    try:
        h, m = str(hhmm).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return default


def in_quiet_hours(now=None) -> bool:
    start = _minutes(_cfg("PHONE_PING_QUIET_START", "23:00"), 23 * 60)
    end = _minutes(_cfg("PHONE_PING_QUIET_END", "07:00"), 7 * 60)
    if start == end:
        return False
    lt = time.localtime(time.time() if now is None else now)
    m = lt.tm_hour * 60 + lt.tm_min
    return (start <= m < end) if start < end else (m >= start or m < end)


def _item_key(who: str, what: str) -> str:
    return (who + "|" + what).lower()[:120]


def announce_line(fresh: list) -> str:
    names = []
    for w in fresh:
        if w["who"] not in names:
            names.append(w["who"])
    if len(names) == 1:
        return f"Sir, {names[0]} is waiting on you."
    more = len(names) - 3
    return (f"Sir, {len(names)} chats are waiting on you: {', '.join(names[:3])}"
            + (f" and {more} more." if more > 0 else "."))


def step(state: dict, now=None) -> str:
    """Fold one poll in; the line to announce, or ``""``. The first poll only
    records what is already waiting (a restart never re-announces it); a new
    item is announced when it is still there on the next poll, outside quiet
    hours, at most one line per GAP_S and MAX_PER_HOUR an hour."""
    now = time.time() if now is None else now
    keyed = {_item_key(w["who"], w["what"]): w for w in waiting_list(state)}
    if not _baselined[0]:
        _baselined[0] = True
        for k in keyed:
            _seen[k] = now
        return ""
    for k in [k for k in _seen if k not in keyed and now - _seen[k] > 86400]:
        _seen.pop(k, None)
    new = [k for k in keyed if k not in _seen]
    ready = [k for k in new if k in _pending]
    _pending.clear()
    _pending.update(new)
    if not ready or in_quiet_hours(now):
        return ""            # quiet hours: still unseen, said when they end
    _sent_at[:] = [t for t in _sent_at if now - t < 3600]
    if (_sent_at and now - _sent_at[-1] < GAP_S) or len(_sent_at) >= MAX_PER_HOUR:
        _pending.update(ready)     # still new: tried again on the next poll
        return ""
    for k in ready:
        _seen[k] = now
    while len(_seen) > SEEN_KEEP:
        _seen.pop(min(_seen, key=_seen.get), None)
    _sent_at.append(now)
    return announce_line([keyed[k] for k in ready])


def _announce(msg: str) -> None:
    try:
        bc = importlib.import_module("bobert_companion")
        fn = getattr(bc, "proactive_announce", None)
        if callable(fn):
            fn(msg, source="agent_desk", dedupe_key="agent_desk:" + msg)
            return
    except Exception:
        pass
    print(f"  [agent-desk] (no announcer) {msg}")


def _poll_loop() -> None:
    if _stop.wait(FIRST_POLL_DELAY_S):
        return
    while True:
        try:
            if announce_on() and enabled():
                state, err = read_state()
                if state is not None and not err:
                    with _lock:
                        msg = step(state)
                    if msg:
                        print(f"  [agent-desk] {msg}")
                        _announce(msg)
        except Exception as e:
            print(f"  [agent-desk] poll error: {e}")
        try:
            gap = max(15.0, float(_cfg("AGENT_DESK_POLL_S", 60.0)))
        except (TypeError, ValueError):
            gap = 60.0
        if _stop.wait(gap):
            return


def register(actions):
    if not enabled():
        return
    actions["agents_status"] = agents_status
    if announce_on() and os.environ.get("JARVIS_NO_SCREEN_READ") != "1":
        threading.Thread(target=_poll_loop, name="agent-desk", daemon=True).start()
        print("  [agent-desk] announcing new waiting-on-you chats (read-only)")
