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


def hotwords_arg(hotwords) -> "str | None":
    """faster-whisper ``hotwords``: a cleaned comma-separated string, or None."""
    if isinstance(hotwords, (list, tuple)):
        hotwords = ", ".join(str(h) for h in hotwords)
    if not isinstance(hotwords, str):
        return None
    words = [w.strip() for w in hotwords.split(",") if w.strip()]
    return ", ".join(words)[:400] or None


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
