"""core/stt_vocab.py — the owner's speech vocabulary (2026-10-01).

Live: "Jarvis, open Accelo on my left monitor" came out of Whisper as "open a cello",
and the turn became a YouTube search for a cello. Two owner settings fix that class:

  * STT_HOTWORDS      — names Whisper should expect (products, people, places):
                        faster-whisper's ``hotwords`` hint, a comma-separated string.
  * STT_REPLACEMENTS  — phrases Whisper keeps mishearing -> what was said, applied to
                        every transcript as whole words, case-insensitively, longest
                        phrase first ({"a cello": "Accelo"}).

And SITE_SHORTCUTS maps a spoken name to a URL ({"accelo": "https://..."}), so "open
Accelo" (or "open Accelo on the left monitor") opens the right page.

All three default to empty in core/config.py, so nothing changes until the owner sets
them in data/user_settings.json. Pure functions; never raise.

STT_HOTWORDS is also read LIVE (live_hotwords, 2026-10-01): core/config.py applies
user_settings.json once at import, so emptying the list after the 22:06 echo drops did
nothing until a restart. Once the file changes after start, its value wins.
"""
from __future__ import annotations

import json
import os
import re
import threading


# Whisper's ``hotwords`` hint is a decoder prompt, and on noise the model can
# "transcribe" the prompt itself. Live 2026-10-01 17:32: with "JARVIS, Accelo, ..." as
# hotwords, a 4 s clip of room noise came back as that very list, and because it began
# with "JARVIS" it passed the wake gate and became a turn. So the wake words are never
# sent as hotwords, and a transcript that is just the list read back is dropped.
_NEVER_HOTWORDS = frozenset({"jarvis", "hey jarvis", "wake", "wake up"})
_ECHO_MIN_HITS = 3          # hotword mentions in the transcript, repeats included
_ECHO_MIN_COVERAGE = 0.6    # share of the transcript's words that are hotwords


def _hotword_list(hotwords) -> "list[str]":
    if isinstance(hotwords, (list, tuple)):
        hotwords = ", ".join(str(h) for h in hotwords)
    if not isinstance(hotwords, str):
        return []
    return [w.strip() for w in hotwords.split(",") if w.strip()]


def hotwords_arg(hotwords) -> "str | None":
    """faster-whisper ``hotwords``: a cleaned comma-separated string without the wake
    words, or None."""
    words = [w for w in _hotword_list(hotwords)
             if " ".join(w.lower().split()) not in _NEVER_HOTWORDS]
    return ", ".join(words)[:400] or None


def is_hotword_echo(text, hotwords) -> bool:
    """True when `text` is Whisper reading the hotwords hint back instead of speech:
    at least three hotword mentions (wake words count here) making up most of it.
    A real request that names a few of them ("open Accelo and Nextcloud") keeps
    enough other words to stay well under the bar.

    Repeats count (2026-10-01): the read-back often repeats one or two names
    ("Accelo, Unraid, Unraid, Unraid", and at 22:13 two names got through as a
    turn), so mentions are counted, not distinct names. At least one mention must
    be a real hotword: the wake words are never sent as hotwords, so "Jarvis,
    Jarvis, Jarvis" is the owner calling, not the hint read back."""
    if not text or not isinstance(text, str):
        return False
    real = {" ".join(w.lower().split()) for w in _hotword_list(hotwords)} - _NEVER_HOTWORDS
    phrases = real | _NEVER_HOTWORDS
    words = re.findall(r"[\w']+", text.lower())
    if not words:
        return False
    rest = " ".join(words)
    hits = covered = real_hits = 0
    for p in sorted(phrases, key=len, reverse=True):   # "hey jarvis" before "jarvis"
        pw = re.findall(r"[\w']+", p)
        if not pw:
            continue
        rest, n = re.subn(r"(?<!\S)" + " ".join(map(re.escape, pw)) + r"(?!\S)", " ", rest)
        if n:
            hits += n
            covered += n * len(pw)
            if p in real:
                real_hits += n
    return (hits >= _ECHO_MIN_HITS and real_hits >= 1
            and covered / len(words) >= _ECHO_MIN_COVERAGE)


# ── STT_HOTWORDS, live (2026-10-01) ─────────────────────────────────────────
# core/config.py reads data/user_settings.json once, at import (the same path it
# uses, so the two agree). LiveSettings remembers that file's mtime at start and
# keeps the import-time value until the file changes; then the file's value wins
# (a removed key means the shipped default, ""). One os.stat per transcription;
# the file is re-parsed only when its mtime moves. A half-written file keeps the
# last good parse. Never raises.
_SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "user_settings.json")


def _mtime_ns(path):
    try:
        return os.stat(path).st_mtime_ns
    except Exception:
        return None


class LiveSettings:
    """The settings file as it is now, for keys that should apply without a
    restart. ``path`` is fixed at construction; its mtime then is the baseline."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._base = _mtime_ns(path)
        self._seen = self._base
        self._data = None          # last good parse after a change
        self._lock = threading.Lock()

    def _current(self):
        """The parsed file when it has changed since start, else None."""
        m = _mtime_ns(self.path)
        if m == self._base:
            return None
        with self._lock:
            if m != self._seen or self._data is None:
                try:
                    with open(self.path, "r", encoding="utf-8-sig") as f:
                        d = json.load(f)
                    if isinstance(d, dict):
                        self._data = d
                        self._seen = m
                    elif self._data is None:
                        self._data = {}
                except Exception:
                    if m is None:              # deleted: nothing set any more
                        self._data, self._seen = {}, m
                    # else half-written: keep the last good parse, re-read next time
            return self._data

    def hotwords(self, fallback):
        """STT_HOTWORDS now: `fallback` (the import-time value) until the file
        changes, then the file's value ("" once the key is removed). A junk
        value (not a string or list) keeps `fallback`."""
        try:
            d = self._current()
            if d is None:
                return fallback
            if "STT_HOTWORDS" not in d:
                return ""
            v = d["STT_HOTWORDS"]
            return v if isinstance(v, (str, list, tuple)) else fallback
        except Exception:
            return fallback


_LIVE = LiveSettings(_SETTINGS_FILE)


def live_hotwords(fallback):
    """STT_HOTWORDS as data/user_settings.json says now (see LiveSettings)."""
    return _LIVE.hotwords(fallback)


def apply_replacements(text: str, mapping) -> str:
    """`text` with each mapped phrase replaced (whole words, case-insensitive, longest
    first so "a cello app" beats "a cello"). Junk entries are ignored."""
    if not text or not isinstance(mapping, dict) or not mapping:
        return text
    pairs = [(str(k).strip(), str(v)) for k, v in mapping.items()
             if isinstance(k, str) and k.strip() and isinstance(v, str)]
    for src, dst in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        pattern = r"(?<![\w'])" + r"\s+".join(re.escape(w) for w in src.split()) + r"(?![\w'])"
        text = re.sub(pattern, dst, text, flags=re.IGNORECASE)
    return text


# "you two" / "u2" / "you tube" is how the STT writes "YouTube" (live
# 2026-10-05 00:32:14: "open YouTube back up" became "opening you two back
# up", and JARVIS opened Chrome AND moved another window). Rewritten ONLY
# inside an open / close / play style command: a verb, up to two filler words,
# then the misheard name - or "play / put on / watch ... on you two". "I'll
# see you two tomorrow" and "you two should talk" are never touched.
# "U2" is a band: it only reads as YouTube after an open / close verb, never
# after play / watch / put on.
_YT_HEARD = r"(?:you\s*-?\s*two|you\s*-?\s*tube|u\s*-?\s*tube)"
_YT_HEARD_OPEN = r"(?:" + _YT_HEARD + r"|u\s*-?\s*2|u\s*-?\s*two)"
_YT_CMD_VERB_RE = (r"(?:open(?:s|ed|ing)?|clos(?:e|es|ed|ing)|launch(?:es|ed|ing)?|"
                   r"start(?:s|ed|ing)?|pull(?:ing)?\s+up|bring(?:ing)?\s+up|"
                   r"go(?:ing)?\s+to|switch(?:ing)?\s+to)")
_YT_PLAY_VERB_RE = r"(?:play(?:s|ed|ing)?|watch(?:es|ed|ing)?|put(?:ting)?\s+on)"
_YT_MISHEARD_RES = (
    re.compile(r"(?<![\w'])(" + _YT_CMD_VERB_RE + r"(?:\s+(?:up|the|a|an|my|our|"
               r"me|some|us))*\s+)" + _YT_HEARD_OPEN + r"(?![\w'])", re.IGNORECASE),
    re.compile(r"(?<![\w'])(" + _YT_PLAY_VERB_RE + r"(?:\s+(?:the|a|an|my|our|"
               r"me|some|us))*\s+)" + _YT_HEARD + r"(?![\w'])", re.IGNORECASE),
    re.compile(r"(?<![\w'])((?:play|plays|playing|put\s+on|watch|watching)\b"
               r"[^.?!]{0,60}?\s(?:on|in|from)\s+)" + _YT_HEARD + r"(?![\w'])",
               re.IGNORECASE),
)


def fix_command_mishearings(text: str) -> str:
    """``text`` with a misheard "YouTube" ("you two", "u2", "you tube")
    restored inside an open / close / play command; everything else, and any
    "you two" outside such a command, unchanged. Never raises."""
    try:
        if not isinstance(text, str) or not text:
            return text
        for rx in _YT_MISHEARD_RES:
            text = rx.sub(lambda m: m.group(1) + "YouTube", text)
        return text
    except Exception:
        return text


_SHORTCUT_NOISE_RE = re.compile(r"^(?:the|my|our)\s+|\s+(?:site|website|page|tickets?|app)$")


def site_shortcut(name, shortcuts) -> "str | None":
    """The URL for a spoken site name ("Accelo", "my accelo tickets"), else None."""
    if not isinstance(shortcuts, dict) or not shortcuts or not isinstance(name, str):
        return None
    s = " ".join(name.lower().split())
    for _ in range(3):
        s2 = _SHORTCUT_NOISE_RE.sub("", s).strip()
        if s2 == s:
            break
        s = s2
    for key, url in shortcuts.items():
        if (isinstance(key, str) and isinstance(url, str)
                and url.startswith(("http://", "https://"))
                and " ".join(key.lower().split()) == s):
            return url
    return None
