"""Read the Windows "now playing" media session (SMTC).

SMTC = System Media Transport Controls, the OS-level now-playing that Chrome,
Spotify, the Apple Music app, YouTube, VLC, etc. all report to (it is what
powers the media flyout shown when you press the keyboard play/pause key).
Reading it is source-agnostic and far more reliable than scraping a browser
window title (which yields the useless "Apple Music: Apple Music").

Backed by the WinRT ``Windows.Media.Control`` projection
(``winrt-Windows.Media.Control``, matched to the installed ``winrt-runtime``).
Everything degrades to ``None`` when winrt / Windows / a live session is absent,
so the tray label and the ``now_playing`` action still render on any machine
(CI / Linux, a box without the projection, or when nothing is playing).

No background thread: ``get_now_playing()`` reads synchronously on demand and
caches the result for ~2s, so the (right-click) tray label and the occasional
voice action stay cheap without leaking a daemon into the test suite.
"""
from __future__ import annotations

import asyncio
import threading
import time

# Seconds a snapshot is reused before the next on-demand SMTC read.
_REFRESH_INTERVAL = 2.0

_snapshot: dict | None = None      # last read {app,title,artist,status,playing} or None
_last_read = 0.0
_lock = threading.Lock()
_available: bool | None = None     # tri-state cache of the winrt import probe

# Windows.Media.Control.GlobalSystemMediaTransportControlsSessionPlaybackStatus
_STATUS_NAMES = {
    0: "closed", 1: "opened", 2: "changing",
    3: "stopped", 4: "playing", 5: "paused",
}


def _winrt_available() -> bool:
    """True iff the SMTC projection imports (probed once, then cached). Tests
    pin ``_available`` directly to stay deterministic across platforms."""
    global _available
    if _available is None:  # pragma: no cover - platform/env dependent
        try:
            from winrt.windows.media.control import (  # noqa: F401
                GlobalSystemMediaTransportControlsSessionManager as _M,
            )
            _available = True
        except Exception:
            _available = False
    return _available


def _clean_app(aumid: str) -> str:
    """Map a raw AppUserModelID to a short, friendly source name."""
    a = (aumid or "").split("!")[0].split("_")[0]
    low = a.lower()
    if "chrome" in low:
        return "Chrome"
    if "msedge" in low or "edge" in low:
        return "Edge"
    if "firefox" in low or "308046b0af4a39cb" in low:
        return "Firefox"
    if "spotify" in low:
        return "Spotify"
    if "applemusic" in low or "apple.music" in low:
        return "Apple Music"
    if "itunes" in low:
        return "iTunes"
    if "vlc" in low:
        return "VLC"
    if "zune" in low or "music" in low:
        return "Media Player"
    return a or "media"


async def _read_session_async() -> "dict | None":  # pragma: no cover - winrt-only
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MGR,
    )
    mgr = await MGR.request_async()
    cur = mgr.get_current_session()
    if cur is None:
        return None
    props = await cur.try_get_media_properties_async()
    pb = cur.get_playback_info()
    status = _STATUS_NAMES.get(int(pb.playback_status), "unknown")
    return {
        "app": _clean_app(cur.source_app_user_model_id),
        "title": (props.title or "").strip(),
        "artist": (props.artist or "").strip(),
        "status": status,
        "playing": status == "playing",
    }


def _default_reader() -> "dict | None":  # pragma: no cover - winrt-only
    """Synchronous one-shot SMTC read on a private event loop (safe to call
    from the tray / action threads, which have no running loop)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_read_session_async())
    finally:
        loop.close()


def _refresh_once(reader=None) -> "dict | None":
    """Read once via ``reader`` (defaults to the real SMTC read) and update the
    cached snapshot + timestamp. Never raises — a failed read clears it."""
    global _snapshot, _last_read
    try:
        snap = (reader or _default_reader)()
    except Exception:
        snap = None
    if snap is not None and not isinstance(snap, dict):
        snap = None
    with _lock:
        _snapshot = snap
        _last_read = time.time()
    return snap


def get_now_playing() -> "dict | None":
    """Best-effort current media session as a dict, or ``None``.

    Keys: ``app``, ``title``, ``artist``, ``status``, ``playing``. Returns a
    copy of the cached snapshot when it is younger than ``_REFRESH_INTERVAL``,
    else does one synchronous SMTC read. ``None`` when winrt is unavailable or
    nothing is playing."""
    if not _winrt_available():
        with _lock:
            return dict(_snapshot) if _snapshot else None
    with _lock:  # pragma: no cover - winrt-only
        fresh = _last_read > 0 and (time.time() - _last_read) < _REFRESH_INTERVAL
        cached = dict(_snapshot) if _snapshot else None
    if fresh:  # pragma: no cover - winrt-only
        return cached
    snap = _refresh_once()  # pragma: no cover - winrt-only
    return dict(snap) if snap else None  # pragma: no cover - winrt-only


def now_playing_text(max_len: int = 60) -> "str | None":
    """One-line ``"Title — Artist"`` (em dash; ``" (paused)"`` suffix when
    paused), or ``None`` when nothing is playing / no title is known."""
    snap = get_now_playing()
    if not snap or not snap.get("title"):
        return None
    title = snap["title"]
    artist = snap.get("artist") or ""
    line = f"{title} — {artist}" if artist else title
    if snap.get("status") == "paused":
        line += " (paused)"
    if len(line) > max_len:
        line = line[: max_len - 1].rstrip() + "…"
    return line
