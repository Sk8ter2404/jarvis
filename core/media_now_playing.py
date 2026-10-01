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
import unicodedata

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
        # 2026-10-01: lets music_status tell the Apple Music web player (a
        # song: artist + album) from a video playing in the same browser.
        "album": (props.album_title or "").strip(),
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


# ─── Transport: pause / play / skip ON a chosen session ────────────────────
#
# 2026-10-01 (B029). pause_music / resume_music / next_song / previous_song
# used to press the OS media keys (playpause / nexttrack / prevtrack). Those
# are blind TOGGLES aimed at whatever Windows considers the "current" session:
# with the Apple Music app merely running, "pause the music" toggled an HBO
# video in Chrome, "pause" on a paused player STARTED it, "resume" on a
# playing one paused it, and "next song" skipped the video's episode. SMTC can
# instead call the idempotent TryPause / TryPlay / TrySkip* on ONE session we
# pick by its real state, so pause can only ever pause and resume can only
# ever resume (the repo's NEVER TOGGLE BLIND rule, applied to transport).

_TRANSPORT_OPS = ("pause", "play", "next", "prev")
# How long an action thread waits for one SMTC transport call.
_TRANSPORT_TIMEOUT_S = 3.0
# Friendly app names (see _clean_app) that are web browsers. A browser
# session is whatever its active media tab is: the Apple Music web player,
# or an HBO / YouTube video.
_BROWSER_APPS = frozenset({"Chrome", "Edge", "Firefox"})
_BROWSER_AUMID_HINTS = ("brave", "opera", "vivaldi", "chromium")
# The Microsoft-Store Apple Music app's friendly name (from _clean_app).
_STORE_APP = "Apple Music"


def _is_browser_app(app) -> bool:
    a = str(app or "")
    return a in _BROWSER_APPS or any(h in a.lower() for h in _BROWSER_AUMID_HINTS)


def _fold(text) -> str:
    """Lower-case, NBSP -> space, invisible format characters (the U+200E
    the web player puts in its tab title) dropped, whitespace collapsed."""
    t = str(text or "").replace("\xa0", " ")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    return " ".join(t.lower().split())


def is_web_player_session(info: dict, web_player_titles=()) -> bool:
    """True when a BROWSER media session is the Apple Music web player (the
    owner's music player), not a video playing in the same browser. Pure.

    2026-10-01 (actions-a review): the transport chooser counted only the
    Store app as "the music", so with the live session shape [Store app
    opened, Chrome playing] "next song" still skipped whatever Chrome played
    (an HBO / YouTube video: the next episode), and a PAUSED web player read
    as "nothing playing". Either piece of evidence is enough:
      * the session publishes a SONG: an artist AND an album (the web player
        sets both in its media metadata; a video publishes no album, and the
        live Chrome video read on 2026-10-01 published neither);
      * its title appears in a live Apple Music web-player window title
        (``web_player_titles``: the tab title carries the playing track).
    """
    if not _is_browser_app(info.get("app")):
        return False
    if (info.get("artist") or "").strip() and (info.get("album") or "").strip():
        return True
    title = _fold(info.get("title"))
    return bool(title) and any(title in _fold(t) for t in web_player_titles or ())


def choose_transport_target(sessions: list, op: str
                            ) -> "tuple[str, int | None]":
    """Pick which session ``op`` acts on. Pure, so it is tested on any OS.

    ``sessions``: one dict per SMTC session, ``{"app", "status", "current",
    "music"}`` (``status`` from ``_STATUS_NAMES``; ``current`` = it is the OS
    current session; ``music`` = ``"web"`` for the Apple Music web player,
    ``"app"`` for the Store app, else falsy). Returns ``(outcome, index)``:
    ``("go", i)`` to act on session ``i``; ``("already", i)`` when the request
    is already true of session ``i`` (pause with nothing playing but
    something paused; resume while something plays); ``("not_music", i)``
    when next/prev would only skip a browser session that isn't the music (a
    video, never skipped on "next song"); or ``("none", None)``.

    The music player always wins: the web player first (the owner's player),
    the Store app only when no web-player session fits; then the OS current
    session; then list order.
    """
    rank = {"web": 0, "app": 1}

    def _best(pred):
        found = [(rank.get(s.get("music"), 2), not s.get("current"), i)
                 for i, s in enumerate(sessions) if pred(s)]
        return min(found)[2] if found else None

    playing = lambda s: s.get("status") == "playing"  # noqa: E731
    paused = lambda s: s.get("status") == "paused"    # noqa: E731
    music = lambda s: s.get("music") in rank           # noqa: E731
    if op == "pause":
        i = _best(playing)
        if i is not None:
            return "go", i
        i = _best(paused)
        return ("already", i) if i is not None else ("none", None)
    if op == "play":
        # Music already playing: never start a second player. A paused music
        # player wins even over another app's video: the owner said "resume
        # the MUSIC". Otherwise resume only when nothing at all is playing.
        i = _best(lambda s: music(s) and playing(s))
        if i is not None:
            return "already", i
        i = _best(lambda s: music(s) and paused(s))
        if i is not None:
            return "go", i
        i = _best(playing)
        if i is not None:
            return "already", i
        i = _best(paused)
        return ("go", i) if i is not None else ("none", None)
    # next / prev: "next SONG" means the music player (a playing one before a
    # paused one), then a playing non-browser player (Spotify, VLC ...).
    i = _best(lambda s: music(s) and playing(s))
    if i is None:
        i = _best(lambda s: music(s) and paused(s))
    if i is None:
        i = _best(lambda s: playing(s) and not _is_browser_app(s.get("app")))
    if i is not None:
        return "go", i
    # Only a browser session that isn't the web player is left: skipping it
    # would skip the video (the next episode). Say so instead.
    i = _best(lambda s: (playing(s) or paused(s)) and _is_browser_app(s.get("app")))
    return ("not_music", i) if i is not None else ("none", None)


async def _transport_async(op: str, web_player_titles):  # pragma: no cover - winrt-only
    """One SMTC transport call -> ``(outcome, app)`` (see transport())."""
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MGR,
    )
    mgr = await MGR.request_async()
    sessions = list(mgr.get_sessions())
    cur = mgr.get_current_session()
    cur_id = cur.source_app_user_model_id if cur is not None else None
    infos = []
    for s in sessions:
        status = _STATUS_NAMES.get(int(s.get_playback_info().playback_status),
                                   "unknown")
        info = {"app": _clean_app(s.source_app_user_model_id),
                "status": status,
                "current": s.source_app_user_model_id == cur_id}
        if _is_browser_app(info["app"]):
            try:
                props = await s.try_get_media_properties_async()
                info.update(title=(props.title or "").strip(),
                            artist=(props.artist or "").strip(),
                            album=(props.album_title or "").strip())
            except Exception:  # noqa: BLE001 - unreadable: not the web player
                pass
        info["music"] = ("app" if info["app"] == _STORE_APP else
                         "web" if is_web_player_session(info, web_player_titles)
                         else None)
        infos.append(info)
    outcome, idx = choose_transport_target(infos, op)
    if idx is None:
        return outcome, None
    app = infos[idx]["app"]
    if outcome != "go":
        return outcome, app
    sess = sessions[idx]
    call = {"pause": sess.try_pause_async, "play": sess.try_play_async,
            "next": sess.try_skip_next_async,
            "prev": sess.try_skip_previous_async}[op]
    ok = await call()
    return ("done" if ok else "failed"), app


def _default_transport(op: str, web_player_titles=()):  # pragma: no cover - winrt-only
    """Run one SMTC transport call on a short-lived worker thread with its own
    event loop, so the WinRT call never runs on the caller's (possibly main)
    thread and a hung call can't freeze it. A timeout reports "failed" — never
    a fallback key press, which could double-act once the slow call lands.

    This really pauses / skips the owner's media, so the test suite's
    hermetic guard (tools/hermetic_guard.py) refuses it: an unpinned test
    can never reach it."""
    box: dict = {}

    def _run():
        loop = asyncio.new_event_loop()
        try:
            box["v"] = loop.run_until_complete(
                _transport_async(op, tuple(web_player_titles or ())))
        except Exception as e:  # noqa: BLE001 - reported to the caller
            box["e"] = e
        finally:
            loop.close()

    t = threading.Thread(target=_run, name="smtc-transport", daemon=True)
    t.start()
    t.join(_TRANSPORT_TIMEOUT_S)
    if "e" in box:
        raise box["e"]
    return box.get("v", ("failed", None))


_TRANSPORT_OUTCOMES = ("done", "already", "none", "failed", "not_music")


def transport(op: str, web_player_titles=(),
              runner=None) -> "tuple[str, str | None] | None":
    """Pause / resume / skip the right media session, idempotently.

    ``op`` is one of ``"pause"``, ``"play"``, ``"next"``, ``"prev"``.
    ``web_player_titles``: the live Apple Music web-player window titles (see
    is_web_player_session). Returns ``(outcome, app)`` with outcome
    ``"done"`` (the session accepted it), ``"already"`` (pause on a paused
    player, resume while one plays), ``"not_music"`` (next/prev with only a
    browser video to skip, left alone), ``"none"`` (no media session fits)
    or ``"failed"`` (the session refused, the call errored or timed out).
    Returns ``None`` ONLY when the SMTC projection is unavailable (CI / Linux
    / no winrt), so the caller may use its legacy path. Never raises.
    ``runner(op, web_player_titles)`` is the test seam."""
    global _last_read
    if op not in _TRANSPORT_OPS:
        return ("failed", None)
    if runner is None:
        if not _winrt_available():
            return None
        runner = _default_transport  # pragma: no cover - winrt-only
    try:
        res = runner(op, tuple(web_player_titles or ()))
    except Exception as e:  # noqa: BLE001 - never raise into an action
        print(f"  [smtc] transport {op} failed: {type(e).__name__}: {e}",
              flush=True)
        return ("failed", None)
    if (not isinstance(res, tuple) or len(res) != 2
            or res[0] not in _TRANSPORT_OUTCOMES):
        return ("failed", None)
    with _lock:  # the cached now-playing snapshot is stale after a transport
        _last_read = 0.0
    return res


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
