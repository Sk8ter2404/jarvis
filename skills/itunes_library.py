"""itunes_library — Siri/Alexa-style control of the local iTunes / Apple Music library.

Adds the library/playlist commands the existing music actions lacked:
  * play_playlist, <name>   — play one of YOUR named playlists ("play my 90s Rock playlist")
  * list_playlists          — read back the playlists you have ("what playlists do I have")
  * shuffle_library         — shuffle-play your whole music library ("shuffle my music")

play_playlist ALWAYS prefers the local iTunes COM route (instant, no fragile
vision-clicking). When the requested name isn't in the local library — or iTunes
itself is unreachable — it AUTOMATICALLY falls back to the monolith's browser
`apple_music` action so an Apple-Music-curated playlist the user doesn't own can
still stream. See `_apple_music_fallback`.

Song / album / artist playback is already handled by the existing `play_music`
action (it understands song:/album:/artist:/library: prefixes); this skill fills
the playlist-by-name gap that `play_music` could not reach.

Built on `audio.itunes_bridge.get_client`, the lazy COM client: it never imports
win32com or launches iTunes at module-import time, and only attaches when iTunes
is already running (unless the user opted into ITUNES_AUTO_LAUNCH). Every COM
touch here is wrapped — any failure degrades to a spoken-friendly line, never an
exception that could crash the action dispatcher.
"""
from __future__ import annotations

import sys

from audio import itunes_bridge

# iTunes COM enums we depend on (stable across iTunes versions):
#   ITPlaylistKind:            Library = 1, User = 2
#   ITUserPlaylistSpecialKind: None = 0 (a real user-created playlist); the auto
#       Music / Movies / TV Shows / Podcasts / Audiobooks / Music Videos lists
#       are all non-zero and are intentionally excluded from "your playlists".
_KIND_USER = 2
_SPECIAL_NONE = 0

# Fold smart punctuation to ASCII so "taylors mix" matches a stored
# "Taylor’s Mix #1" (iTunes stores curly apostrophes/quotes).
_SMART_PUNCT = (
    ("’", "'"), ("‘", "'"), ("ʼ", "'"),
    ("“", '"'), ("”", '"'),
    ("–", "-"), ("—", "-"),
)


def _norm(s: str) -> str:
    """Lowercase, fold smart quotes/dashes, DROP apostrophes (so a spoken
    'taylors mix' matches a stored 'Taylor's Mix'), collapse whitespace."""
    s = (s or "").strip().lower()
    for a, b in _SMART_PUNCT:
        s = s.replace(a, b)
    s = s.replace("'", "")
    return " ".join(s.split())


def _user_playlists(app):
    """Return [(playlist_com_obj, name), ...] for real USER playlists only —
    skips the top-level Library and the auto Music/Movies/Podcasts/... lists."""
    out = []
    pls = app.LibrarySource.Playlists
    for i in range(1, int(pls.Count) + 1):
        try:
            pl = pls.Item(i)
            if int(getattr(pl, "Kind", 0)) != _KIND_USER:
                continue
            if int(getattr(pl, "SpecialKind", _SPECIAL_NONE)) != _SPECIAL_NONE:
                continue
            name = getattr(pl, "Name", "") or ""
            if name:
                out.append((pl, name))
        except Exception:
            continue
    return out


def _match_playlist(playlists, query):
    """Best name match over (obj, name) pairs: exact > startswith > substring."""
    q = _norm(query)
    if not q:
        return None, None
    for pl, n in playlists:
        if _norm(n) == q:
            return pl, n
    for pl, n in playlists:
        if _norm(n).startswith(q):
            return pl, n
    for pl, n in playlists:
        if q in _norm(n):
            return pl, n
    return None, None


def _strip_shuffle(text):
    """Detect a LEADING 'shuffle'/'shuffled' keyword. Returns (clean_name, bool).
    Only leading is honoured — a trailing match would wrongly strip a playlist
    literally named e.g. 'Evening Shuffle'."""
    t = (text or "").strip()
    low = t.lower()
    for kw in ("shuffle ", "shuffled "):
        if low.startswith(kw):
            return t[len(kw):].strip(), True
    return t, False


def _apple_music_fallback(query: str) -> str | None:
    """Last-resort route for a playlist NOT in the local iTunes library: hand
    the request to the monolith's browser-based `apple_music` action so an
    Apple-Music-curated playlist (one the user doesn't own) can still stream.

    Reaches the running monolith via sys.modules (it runs as __main__, or is
    importable as bobert_companion) and calls ACTIONS["apple_music"](query).
    Returns the action's spoken string, or None if the monolith / action isn't
    reachable for any reason — every failure degrades to None, never raises, so
    play_playlist can fall through to its own not-found message."""
    try:
        bc = sys.modules.get("__main__") or sys.modules.get("bobert_companion")
        actions = getattr(bc, "ACTIONS", None)
        if actions and "apple_music" in actions:
            result = actions["apple_music"](query)
            return result if isinstance(result, str) else None
    except Exception:
        return None
    return None


def play_playlist(arg: str) -> str:
    name_q, shuffle = _strip_shuffle(arg)
    if not name_q:
        return "Which playlist would you like, sir?"
    app, err = itunes_bridge.get_client(force=True)
    if app is None:
        # iTunes COM unreachable → try streaming the playlist on Apple Music.
        fb = _apple_music_fallback(name_q + (" shuffle" if shuffle else ""))
        if fb:
            return fb
        return err or "iTunes isn't reachable right now, sir."
    try:
        pl, name = _match_playlist(_user_playlists(app), name_q)
    except Exception as e:
        return f"I couldn't read your playlists, sir: {e}"
    if pl is None:
        # Not one of the user's OWN playlists → fall through to Apple Music
        # streaming (e.g. a curated playlist they don't own). The apple_music
        # action speaks its own sensible line, so return it as-is.
        fb = _apple_music_fallback(name_q + (" shuffle" if shuffle else ""))
        if fb:
            return fb
        return f"I couldn't find a playlist called '{name_q}', sir."
    try:
        if shuffle:
            try:
                pl.Shuffle = True
            except Exception:
                pass
        pl.PlayFirstTrack()
    except Exception as e:
        return f"I found your '{name}' playlist but couldn't start it, sir: {e}"
    return f"Playing your '{name}' playlist{' shuffled' if shuffle else ''}, sir."


def list_playlists(arg: str = "") -> str:
    app, err = itunes_bridge.get_client(force=True)
    if app is None:
        return err or "iTunes isn't reachable right now, sir."
    try:
        names = [n for _, n in _user_playlists(app)]
    except Exception as e:
        return f"I couldn't read your playlists, sir: {e}"
    if not names:
        return "I don't see any custom playlists in your iTunes library, sir."
    total = len(names)
    shown = names[:20]
    listing = ", ".join(shown)
    if total > len(shown):
        listing += f", and {total - len(shown)} more"
    return f"You have {total} playlists, sir: {listing}."


def shuffle_library(arg: str = "") -> str:
    """Shuffle-play the whole music library (the auto 'Music' playlist)."""
    app, err = itunes_bridge.get_client(force=True)
    if app is None:
        return err or "iTunes isn't reachable right now, sir."
    try:
        target = None
        pls = app.LibrarySource.Playlists
        for i in range(1, int(pls.Count) + 1):
            try:
                pl = pls.Item(i)
                if (getattr(pl, "Name", "") or "").lower() == "music":
                    target = pl
                    break
            except Exception:
                continue
        if target is None:
            target = app.LibraryPlaylist
        try:
            target.Shuffle = True
        except Exception:
            pass
        target.PlayFirstTrack()
    except Exception as e:
        return f"I couldn't shuffle your library, sir: {e}"
    return "Shuffling your music library, sir."


def register(actions: dict) -> None:
    actions["play_playlist"] = play_playlist
    actions["list_playlists"] = list_playlists
    actions["shuffle_library"] = shuffle_library
