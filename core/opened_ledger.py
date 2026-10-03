"""core/opened_ledger.py - what JARVIS itself opened last, and "close that".

WHY THIS EXISTS (2026-10-02)
============================
Live: JARVIS had just put a show's search page on the main monitor. The owner
said "close that and open <the streaming service> instead". The brain answered
with ONE token - the open - and the close was dropped. Nothing caught it:

  * core.dispatcher.command_chain_resolver only claims a turn when two or more
    segments match its own rules; it has no "close" or "open" rule, so the
    turn went to the brain;
  * the monolith's continuation enforcer (_detect_dropped_steps) reads only
    the MODEL's prose for promised future steps ("I'll read it to you"); it
    never looks at the owner's own compound command, and it had no close
    intent at all;
  * and even a close the brain DID emit would have been close_window with a
    guessed title, which can match one of the owner's own windows.

Two pieces fix that:

  * a LEDGER of the windows and tabs JARVIS opened (open_on_monitor, open_url,
    web_search, the streaming actions record into it), so "that" means the
    last thing JARVIS opened - never anything else;
  * ``is_close_then_open`` + ``rewrite_close_then_open``: when the owner's words
    are "close that ... and open ..." and the reply opens something without
    closing, ``[ACTION: close_last_opened]`` is put in front of the open, and
    any close the brain guessed at (close_window, close_app ...) is replaced by
    it - the owner said "that", he did not name a window.

Pure stdlib; thread-safe; never raises.
"""
from __future__ import annotations

import re
import threading
import time
from typing import NamedTuple, Optional

from core.lead_fillers import strip_lead_filler as _strip_lead_filler

# How long an opened window stays "that" for close_last_opened.
CLOSE_MAX_AGE_S = 30 * 60.0
# How long the page JARVIS opened decides which monitor a click / a look at
# "the page" is aimed at when the owner names none (S4).
PAGE_MAX_AGE_S = 10 * 60.0
_MAX_ENTRIES = 8


class Opened(NamedTuple):
    via: str                 # the action that opened it
    target: str              # the URL or app name
    hwnd: Optional[int]      # the window, when known
    kind: str                # "window" (JARVIS made it) | "tab" (a tab it added)
    monitor: Optional[str]   # the MONITORS key it is on, when known
    title: str               # the window title when recorded ("" unknown)
    at: float                # time.time()


_lock = threading.Lock()
_entries: list = []          # oldest first


def note_opened(via, target, *, hwnd=None, kind="window", monitor=None,
                title="", now=None) -> Optional[Opened]:
    """Record a window / tab JARVIS just opened. A second record of the same
    window replaces the first. Returns the entry, or None. Never raises."""
    try:
        entry = Opened(str(via or ""), str(target or ""),
                       int(hwnd) if isinstance(hwnd, int) else None,
                       "tab" if kind == "tab" else "window",
                       str(monitor) if monitor else None,
                       str(title or ""),
                       float(time.time() if now is None else now))
        with _lock:
            if entry.hwnd is not None and entry.kind == "window":
                _entries[:] = [e for e in _entries
                               if not (e.hwnd == entry.hwnd
                                       and e.kind == "window")]
            _entries.append(entry)
            del _entries[:-_MAX_ENTRIES]
        return entry
    except Exception:
        return None


def last_opened(max_age_s: float = CLOSE_MAX_AGE_S, now=None) -> Optional[Opened]:
    """The newest entry no older than ``max_age_s``, else None."""
    try:
        t = float(time.time() if now is None else now)
        with _lock:
            for e in reversed(_entries):
                if t - e.at <= max_age_s:
                    return e
                break
    except Exception:
        return None
    return None


def forget(entry) -> None:
    """Drop ``entry`` (an Opened, or a bare hwnd)."""
    try:
        with _lock:
            if isinstance(entry, Opened):
                _entries[:] = [e for e in _entries if e != entry]
            else:
                _entries[:] = [e for e in _entries if e.hwnd != entry]
    except Exception:
        pass


def reset() -> None:
    """Forget everything (tests)."""
    with _lock:
        _entries.clear()


_WEB_TARGET_RE = re.compile(r"^[\w-]+(?:\.[\w-]+)*\.([A-Za-z]{2,24})(?:[/?#]|$)")
# "notepad.exe" names an app, not a site.
_FILE_EXTENSIONS = frozenset({
    "exe", "lnk", "bat", "cmd", "msc", "msi", "ps1", "py", "txt", "pdf",
    "docx", "xlsx", "pptx", "png", "jpg", "jpeg", "mp3", "mp4", "url",
})


def is_web_target(target) -> bool:
    """True when an entry's target is a web page (a URL or a bare host such
    as "max.com"), not an app name ("notepad", "notepad.exe"). Never
    raises."""
    try:
        t = str(target or "").strip()
        if "://" in t:
            return True
        m = _WEB_TARGET_RE.match(t)
        return bool(m) and m.group(1).lower() not in _FILE_EXTENSIONS
    except Exception:
        return False


def describe(entry) -> str:
    """A short spoken name for an entry: "YouTube page", "Notepad window"."""
    try:
        from urllib.parse import urlsplit
        t = str(entry.target or "")
        host = ""
        if is_web_target(t):
            host = (urlsplit(t if "://" in t else "https://" + t).hostname
                    or "")
        if host:
            try:
                from core.streaming_search import service_for_url, service_name
                key = service_for_url(t)
                if key:
                    return f"{service_name(key)} page"
            except Exception:
                pass
            parts = [p for p in host.split(".") if p not in ("www", "m")]
            label = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
            return f"{label.capitalize()} page" if label else "page"
        return f"{t} window" if t else "window"
    except Exception:
        return "window"


# ── "close that ... and open ..." ────────────────────────────────────────────
# "close that" / "close it out" / "close the last one": the deictic forms. A
# NAMED close ("close Spotify and open Netflix") is the owner naming it - the
# brain's close_window stays.
_DEICTIC = (r"(?:that|it|this|those|them|these|"
            r"the\s+(?:last\s+|other\s+|current\s+)?(?:one|window|tab|page|"
            r"video|show|thing|search|results?))")
_CLOSE_RE = re.compile(
    r"\bclose\s+(?:out\s+|down\s+)?" + _DEICTIC +
    r"(?:\s+(?:one|window|tab|page|out|down|up))?\b", re.IGNORECASE)
# Parakeet writes the imperative "close" as "closed" right after the wake
# word ("Jarvis closed that and open ..."); only accepted at the very start.
_CLOSED_LEAD_RE = re.compile(r"^closed\s+" + _DEICTIC + r"\b", re.IGNORECASE)
_OPEN_AFTER_RE = re.compile(
    r"(?:[.;!,]\s*|\s+)(?:(?:and|then|now|please|instead|also)[\s,]+)*"
    r"(?:open|put|pull\s+up|bring\s+up|launch|start|play|show|load|"
    r"go\s+to|navigate\s+to|switch\s+to|search|find)\b", re.IGNORECASE)
_OPEN_WINDOW_CHARS = 60


def is_close_then_open(text) -> bool:
    """True when the owner's words close something deictic ("close that",
    "close it") and then open something in the same breath. Never raises."""
    try:
        s = _strip_lead_filler(" ".join(str(text or "").split()))
        m = _CLOSED_LEAD_RE.match(s) or _CLOSE_RE.search(s)
        if not m:
            return False
        rest = s[m.end():m.end() + _OPEN_WINDOW_CHARS]
        return bool(_OPEN_AFTER_RE.match(rest))
    except Exception:
        return False


# Action names that OPEN something the owner will look at.
OPEN_ACTIONS = frozenset({
    "open_url", "open_on_monitor", "web_search", "youtube", "youtube_play",
    "youtube_search_direct", "play_streaming", "streaming_search", "netflix",
    "max", "hulu", "disney_plus", "prime_video", "apple_music", "spotify",
    "launch_app", "browser_open", "play_music", "play_playlist",
    "open_apple_music",
})
# Closes the brain may guess at for "close that": replaced by
# close_last_opened, because the owner did not name a window. The bulk close
# too (2026-10-03: the real one is close_all_windows_except): "close that"
# never asks for every other window to go.
GUESSED_CLOSE_ACTIONS = frozenset({
    "close_window", "close_tab", "close_app", "close_all_windows",
    "close_all_windows_except",
})
_TOKEN_RE = re.compile(r"\[ACTION:\s*([a-z0-9_]+)\s*(?:,\s*(.+?))?\s*\]",
                       re.IGNORECASE)
CLOSE_ACTION_TAG = "[ACTION: close_last_opened]"


def rewrite_close_then_open(reply) -> tuple:
    """(reply, dropped) for a "close that and open X" turn.

    When ``reply`` opens something (an OPEN_ACTIONS token) and does not
    already run close_last_opened, CLOSE_ACTION_TAG goes right before the
    first open, and every GUESSED_CLOSE_ACTIONS token is taken out
    (``dropped`` lists them). A reply that only closes has its guessed
    close(s) replaced by CLOSE_ACTION_TAG (the open is then the follow-up
    round's dropped step). A reply that neither opens nor closes comes back
    unchanged. Never raises."""
    try:
        text = str(reply or "")
        toks = list(_TOKEN_RE.finditer(text))
        names = [m.group(1).strip().lower() for m in toks]
        if "close_last_opened" in names:
            return text, []
        first_open = next((m for m, n in zip(toks, names)
                           if n in OPEN_ACTIONS), None)
        if first_open is None:
            first_open = next((m for m, n in zip(toks, names)
                               if n in GUESSED_CLOSE_ACTIONS), None)
            if first_open is None:
                return text, []
            dropped = [m.group(0) for m, n in zip(toks, names)
                       if n in GUESSED_CLOSE_ACTIONS]
            out, last, done = [], 0, False
            for m, n in zip(toks, names):
                if n in GUESSED_CLOSE_ACTIONS:
                    out.append(text[last:m.start()])
                    if not done:
                        out.append(CLOSE_ACTION_TAG)
                        done = True
                    last = m.end()
            out.append(text[last:])
            return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip(), dropped
        dropped = [m.group(0) for m, n in zip(toks, names)
                   if n in GUESSED_CLOSE_ACTIONS]
        out, last = [], 0
        for m, n in zip(toks, names):
            if m is first_open:
                out.append(text[last:m.start()])
                out.append(CLOSE_ACTION_TAG + " ")
                out.append(m.group(0))
                last = m.end()
            elif n in GUESSED_CLOSE_ACTIONS:
                out.append(text[last:m.start()])
                last = m.end()
        out.append(text[last:])
        new = re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
        return new, dropped
    except Exception:
        return str(reply or ""), []
