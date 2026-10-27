"""'Play my song from earlier' replays the owner's last media request.

Live 2026-10-26 20:08: "Jarvis, just play my damn song from earlier." - three
minutes after "play famous last words on YouTube on the left monitor" - and
the brain emitted [ACTION: play_music, song:/earlier]: Apple Music played a
track literally titled "Earlier". The replay deque (_action_history) keeps
only the last 5 actions, so the screenshots and searches in between had
already pushed the request out, and nothing else remembered it.

record() keeps the last media REQUEST (a play / a YouTube search opened on a
monitor), with its title and monitor. recall_route() claims a whole "play my
song from earlier / play that song again / play it again" and rebuilds the
request as action tokens for the monolith's built-in route: a YouTube search
becomes youtube_play (the owner asked to PLAY it), plus the monitor move when
one was named. Requests older than RECALL_MAX_AGE_S are forgotten.

Pure: stdlib only, never raises.
"""
from __future__ import annotations

import re
import threading
import time

from core.lead_fillers import strip_lead_filler as _strip_lead_filler

RECALL_MAX_AGE_S = 6 * 3600.0

# Actions whose argument IS the thing to play again.
PLAY_ACTIONS = frozenset({
    "play_music", "apple_music", "spotify", "youtube_play", "play_streaming",
    "youtube_search_direct", "youtube_direct", "yt_direct",
})
# Searches the owner meant as "play this": replayed as youtube_play.
YOUTUBE_SEARCH_ACTIONS = frozenset({"youtube"})

_lock = threading.Lock()
_last: list = [None]

_WAKE_RE = re.compile(r"^\s*(?:(?:hey|ok|okay)[\s,]+)?jarvis\b[\s,.:;!?-]*",
                      re.IGNORECASE)
_RECALL_RE = re.compile(
    r"^(?:just\s+)?(?:play|put\s+on|start|replay)\s+(?:back\s+)?"
    r"(?:(?:my|the|that|this)\s+(?:damn\s+|same\s+|last\s+)?"
    r"(?:song|track|video|music|tune)(?:\s+again)?"
    r"(?:\s+(?:from\s+)?(?:earlier|before|from\s+before|"
    r"(?:that\s+|which\s+)?i\s+(?:asked\s+for|wanted|had\s+on|played)))?"
    r"|it\s+again|that\s+again)"
    r"[\s.!?]*$", re.IGNORECASE)
_MONITOR_ARG_RE = re.compile(r"^\s*([a-z]+)\s*\|\s*youtube\b\s*(.*)$",
                             re.IGNORECASE)


def _clean_title(text: str) -> str:
    t = (text or "").strip().strip('"').strip("'").strip()
    t = re.sub(r"^(?:song|track|video)\s*:\s*/?", "", t, flags=re.IGNORECASE)
    return " ".join(t.split())


def request_from(action: str, arg: str):
    """{"action", "title", "monitor"} for a media request, else None."""
    try:
        name = (action or "").strip().lower()
        a = str(arg or "")
        if name in PLAY_ACTIONS:
            title = _clean_title(a)
            return {"action": name, "title": title, "monitor": ""} if title else None
        if name in YOUTUBE_SEARCH_ACTIONS:
            title = _clean_title(a)
            return ({"action": "youtube_play", "title": title, "monitor": ""}
                    if title else None)
        if name == "open_on_monitor":
            m = _MONITOR_ARG_RE.match(a)
            if m and _clean_title(m.group(2)):
                return {"action": "youtube_play", "title": _clean_title(m.group(2)),
                        "monitor": m.group(1).lower()}
        return None
    except Exception:
        return None


def record(action: str, arg: str, now: float | None = None) -> None:
    """Remember this action when it is a media request. Never raises."""
    req = request_from(action, arg)
    if not req:
        return
    req["at"] = time.time() if now is None else float(now)
    with _lock:
        _last[0] = req


def last_request(now: float | None = None):
    now = time.time() if now is None else now
    with _lock:
        req = dict(_last[0]) if _last[0] else None
    if not req or now - float(req.get("at") or 0.0) > RECALL_MAX_AGE_S:
        return None
    return req


def is_recall_request(utterance) -> bool:
    try:
        s = _WAKE_RE.sub("", str(utterance or ""), count=1)
        s = _strip_lead_filler(s.strip())
        return bool(_RECALL_RE.match(" ".join(s.split())))
    except Exception:
        return False


def recall_route(utterance, now: float | None = None):
    """Action tokens replaying the last media request for a whole "play my
    song from earlier"; None when the words are something else or nothing
    recent was requested (the model keeps those). Never raises."""
    try:
        if not is_recall_request(utterance):
            return None
        req = last_request(now)
        if not req or any(c in req["title"] for c in "[]\r\n"):
            return None
        out = f"[ACTION: {req['action']}, {req['title']}]"
        if req.get("monitor"):
            out += f" [ACTION: move_window_to_monitor, youtube | {req['monitor']}]"
        return out
    except Exception:
        return None


def reset() -> None:
    """Forget the last request (tests)."""
    with _lock:
        _last[0] = None
