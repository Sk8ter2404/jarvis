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
"""
from __future__ import annotations

import re


# Whisper's ``hotwords`` hint is a decoder prompt, and on noise the model can
# "transcribe" the prompt itself. Live 2026-10-01 17:32: with "JARVIS, Accelo, ..." as
# hotwords, a 4 s clip of room noise came back as that very list, and because it began
# with "JARVIS" it passed the wake gate and became a turn. So the wake words are never
# sent as hotwords, and a transcript that is just the list read back is dropped.
_NEVER_HOTWORDS = frozenset({"jarvis", "hey jarvis", "wake", "wake up"})
_ECHO_MIN_HITS = 3          # distinct hotwords in the transcript
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
    at least three distinct hotwords (wake words count here) making up most of it.
    A real request that names a few of them ("open Accelo and Nextcloud") keeps
    enough other words to stay well under the bar."""
    if not text or not isinstance(text, str):
        return False
    phrases = {" ".join(w.lower().split()) for w in _hotword_list(hotwords)} | _NEVER_HOTWORDS
    words = re.findall(r"[\w']+", text.lower())
    if not words:
        return False
    rest = " ".join(words)
    hits = covered = 0
    for p in sorted(phrases, key=len, reverse=True):   # "hey jarvis" before "jarvis"
        pw = re.findall(r"[\w']+", p)
        if not pw:
            continue
        rest, n = re.subn(r"(?<!\S)" + " ".join(map(re.escape, pw)) + r"(?!\S)", " ", rest)
        if n:
            hits += 1
            covered += n * len(pw)
    return hits >= _ECHO_MIN_HITS and covered / len(words) >= _ECHO_MIN_COVERAGE


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
