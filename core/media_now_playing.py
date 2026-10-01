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


def choose_transport_target(sessions: list, op: str,
                            prefer_app: str = "Apple Music"
                            ) -> "tuple[str, int | None]":
    """Pick which session ``op`` acts on. Pure, so it is tested on any OS.

    ``sessions``: one dict per SMTC session, ``{"app", "status", "current"}``
    (``status`` from ``_STATUS_NAMES``; ``current`` = it is the OS current
    session). Returns ``(outcome, index)``: ``("go", i)`` to act on session
    ``i``, ``("already", i)`` when the request is already true of session
    ``i`` (pause with nothing playing but something paused; resume while
    something plays), or ``("none", None)`` when no session fits at all.

    Preference order, everywhere: the ``prefer_app`` session, then the OS
    current session, then the first that fits.
    """
    def _first(pred):
        for want in (lambda s: s.get("app") == prefer_app,
                     lambda s: bool(s.get("current")),
                     lambda s: True):
            for i, s in enumerate(sessions):
                if pred(s) and want(s):
                    return i
        return None

    playing = lambda s: s.get("status") == "playing"  # noqa: E731
    paused = lambda s: s.get("status") == "paused"    # noqa: E731
    if op == "pause":
        i = _first(playing)
        if i is not None:
            return "go", i
        i = _first(paused)
        return ("already", i) if i is not None else ("none", None)
    if op == "play":
        # A paused preferred player wins even over another app's video: the
        # owner said "resume the MUSIC". Otherwise never start a second,
        # arbitrary player while something is already playing.
        for i, s in enumerate(sessions):
            if s.get("app") == prefer_app and paused(s):
                return "go", i
        i = _first(playing)
        if i is not None:
            return "already", i
        i = _first(paused)
        return ("go", i) if i is not None else ("none", None)
    # next / prev: "next SONG" means the music player even while it is
    # paused; otherwise only a session that is actually playing.
    for i, s in enumerate(sessions):
        if s.get("app") == prefer_app and (playing(s) or paused(s)):
            return "go", i
    i = _first(playing)
    return ("go", i) if i is not None else ("none", None)


async def _transport_async(op: str, prefer_app: str):  # pragma: no cover - winrt-only
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
        infos.append({"app": _clean_app(s.source_app_user_model_id),
                      "status": status,
                      "current": s.source_app_user_model_id == cur_id})
    outcome, idx = choose_transport_target(infos, op, prefer_app)
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


def _default_transport(op: str, prefer_app: str):  # pragma: no cover - winrt-only
    """Run one SMTC transport call on a short-lived worker thread with its own
    event loop, so the WinRT call never runs on the caller's (possibly main)
    thread and a hung call can't freeze it. A timeout reports "failed" — never
    a fallback key press, which could double-act once the slow call lands."""
    box: dict = {}

    def _run():
        loop = asyncio.new_event_loop()
        try:
            box["v"] = loop.run_until_complete(_transport_async(op, prefer_app))
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


def transport(op: str, prefer_app: str = "Apple Music",
              runner=None) -> "tuple[str, str | None] | None":
    """Pause / resume / skip the right media session, idempotently.

    ``op`` is one of ``"pause"``, ``"play"``, ``"next"``, ``"prev"``. Returns
    ``(outcome, app)`` with outcome ``"done"`` (the session accepted it),
    ``"already"`` (pause on a paused player, resume while one plays),
    ``"none"`` (no media session fits) or ``"failed"`` (the session refused,
    the call errored or timed out). Returns ``None`` ONLY when the SMTC
    projection is unavailable (CI / Linux / no winrt), so the caller may use
    its legacy path. Never raises. ``runner`` is the test seam."""
    global _last_read
    if op not in _TRANSPORT_OPS:
        return ("failed", None)
    if runner is None:
        if not _winrt_available():
            return None
        runner = _default_transport  # pragma: no cover - winrt-only
    try:
        res = runner(op, prefer_app)
    except Exception as e:  # noqa: BLE001 - never raise into an action
        print(f"  [smtc] transport {op} failed: {type(e).__name__}: {e}",
              flush=True)
        return ("failed", None)
    if (not isinstance(res, tuple) or len(res) != 2
            or res[0] not in ("done", "already", "none", "failed")):
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
