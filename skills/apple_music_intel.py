"""
apple_music_intel — listening-history tracker, taste-pattern learner, and
session-skip memory for Apple Music + iTunes playback.

Why a separate skill rather than another bolt-on to bobert_companion's
existing music actions:
  • The existing _act_play_music / _act_apple_music actions are pure
    dispatchers — they have no notion of "what got played", let alone
    "what got skipped after 8 seconds" or "what artist dominates Friday
    nights". We want all of that without entangling the core music
    plumbing.
  • Pattern_learning already tracks ACTION events. That's the right layer
    for "the user runs play_music at 09:15 daily." It can't see SONG-level
    granularity (which artist, which genre, how long it played) — that's
    what this skill adds.

Data sources (best-effort fallback chain, polled every POLL_SECONDS):
  1. iTunes COM CurrentTrack (preferred — gives Name/Artist/Album/Genre
     plus PlayedCount and authoritative PlayedDate after each play).
  2. Apple Music web UI window title via pygetwindow ("Song — Artist –
     Apple Music"). Used when iTunes COM is offline but the user is
     playing in the browser.
  3. Spotify web UI window title ("Song · Artist - Spotify Web Player").
     Same fallback role.

Storage (all atomic temp+rename, all under data/):
  data/apple_music_history.jsonl    — append-only listen log
  data/apple_music_skips.jsonl      — append-only skip log
  data/apple_music_session.json     — current session's skipped-artist list
                                       (cleared at JARVIS startup)
  data/apple_music_taste.json       — aggregated taste snapshot (artist by
                                       day/hour bucket, skip rate by genre)

Actions registered:
  play_unheard            — play an iTunes track unheard in N+ days
                            (defaults to UNHEARD_MIN_DAYS). Arg: optional
                            integer days override, e.g. "30".
  play_vibe, <slot>       — play the dominant artist for that day/time
                            slot. Slot phrases: 'friday night',
                            'sunday morning', 'now', 'current', or any
                            "<day> <part-of-day>" combo. 'now' / no arg
                            uses the current weekday + time-of-day.
  skip_track              — record current track to session skip memory,
                            log it, then media_next + (best-effort)
                            iTunes NextTrack. The LLM should route
                            "skip — I'm not feeling this one" here so we
                            remember the skip; plain "skip" / "next song"
                            still goes to media_next / next_song.
  music_history           — short readback of the last 5 listens.
  music_taste             — short readback of dominant patterns: top
                            artists overall, top artist this slot, skip
                            rate by genre.
  music_aggregate         — force the taste aggregation to rerun now.

Background:
  • _listen_loop runs every POLL_SECONDS. When the current track changes,
    it emits a 'listen' event with the elapsed-on-previous-track duration.
    A track that played for less than SKIP_THRESHOLD_SECS before changing
    is recorded as a SKIP (not a complete listen).
  • _aggregator_loop rebuilds data/apple_music_taste.json once an hour
    (cheap; mostly counters) so play_vibe / music_taste stay responsive.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict

# Direct import of the iTunes COM bridge — the bridge itself is import-cheap
# (no win32com / pythoncom at module-load) and lazy-gates by is_running() so
# this still doesn't pop iTunes open at boot. Replaces the previous dynamic
# `sys.modules.get("__main__")._get_itunes` lookup, which was fragile to
# import-path differences between __main__-launch and load_skills().
from audio import itunes_bridge  # type: ignore


# ─── paths ────────────────────────────────────────────────────────────────

_HERE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_HERE, "data")
_HISTORY_FILE  = os.path.join(_DATA_DIR, "apple_music_history.jsonl")
_SKIPS_FILE    = os.path.join(_DATA_DIR, "apple_music_skips.jsonl")
_SESSION_FILE  = os.path.join(_DATA_DIR, "apple_music_session.json")
_TASTE_FILE    = os.path.join(_DATA_DIR, "apple_music_taste.json")


# ─── tunables ─────────────────────────────────────────────────────────────

POLL_SECONDS        = 12       # how often to sample the current track
SKIP_THRESHOLD_SECS = 25       # < this much listen time before track-change = skip
UNHEARD_MIN_DAYS    = 14       # default "haven't heard in a while" threshold
MAX_HISTORY_LINES   = 25000    # rotation cap on the jsonl logs
AGGREGATE_INTERVAL  = 3600     # rebuild taste snapshot every hour
INITIAL_DELAY_SECS  = 8        # let JARVIS finish booting before we touch COM
SESSION_SKIP_CAP    = 200      # max artists we keep in the per-session skip set
MIN_LISTENS_FOR_VIBE = 2       # need ≥2 listens in a slot before suggesting it

# Time-of-day buckets used for vibe lookups. Hour ranges are half-open: a
# slot covers [start, end). 'late_night' wraps midnight.
_TIME_OF_DAY = [
    ("morning",     6, 12),
    ("afternoon",   12, 17),
    ("evening",     17, 22),
    ("night",       22, 30),  # 22:00–05:59 next day; modular handling below
]

_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
         "saturday", "sunday")

# Words that count as part-of-day or day-of-week when parsing a vibe slot.
_DAY_ALIASES = {
    "mon": "monday", "tue": "tuesday", "tues": "tuesday",
    "wed": "wednesday", "thu": "thursday", "thur": "thursday",
    "thurs": "thursday", "fri": "friday", "sat": "saturday",
    "sun": "sunday",
}
_PARTS = {
    "morning": "morning", "noon": "afternoon", "midday": "afternoon",
    "afternoon": "afternoon", "evening": "evening", "tonight": "night",
    "night": "night", "late": "night", "lunch": "afternoon",
    "breakfast": "morning",
}


# ─── module state ─────────────────────────────────────────────────────────

_listen_lock = threading.Lock()
# Track currently observed by the background poller. None until first sample.
# Shape: {"key": "<artist>|<title>", "artist": ..., "title": ..., "album": ...,
#         "genre": ..., "source": "itunes"|"web_apple"|"web_spotify",
#         "since": <epoch>, "logged": False}
_current: dict | None = None

_aggregator_started = [False]


# ─── small helpers ────────────────────────────────────────────────────────

def _ensure_data_dir() -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
    except Exception:
        pass


def _atomic_write_json(path: str, payload) -> None:
    _ensure_data_dir()
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _append_jsonl(path: str, entry: dict) -> None:
    _ensure_data_dir()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [apple_music_intel] append failed ({path}): {e}")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def _maybe_rotate(path: str, cap: int = MAX_HISTORY_LINES) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= cap:
            return
        keep = lines[-cap:]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, path)
    except Exception:
        pass


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _track_key(artist: str, title: str) -> str:
    return f"{_norm(artist)}|{_norm(title)}"


def _time_of_day(hour: int) -> str:
    if hour < 0:
        hour = 0
    h = hour % 24
    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"   # 22:00–05:59


def _slot_key(day: str, part: str) -> str:
    return f"{day}|{part}"


# ─── data sources ─────────────────────────────────────────────────────────

def _sample_itunes() -> dict | None:
    """Return a dict describing the current iTunes track, or None if iTunes
    isn't reachable or nothing is playing. Best-effort — exceptions become
    None so the fallback chain can try the browser title next.

    Hard gate: if iTunes.exe isn't already running we return None immediately
    without ever calling into win32com. This keeps the background listen
    loop from initialising iTunes COM at boot — the previous behaviour was
    to call _get_itunes every POLL_SECONDS=12s, which (pre-fix) could spawn
    iTunes via Dispatch and (post-fix) still does a needless CoInitialize +
    GetActiveObject round-trip every poll. itunes_bridge.get_client() also
    enforces this gate internally; the explicit is_running() check here lets
    us skip even the bridge call when there's nothing to bind to."""
    if not itunes_bridge.is_running():
        return None
    # itunes_bridge.get_client() itself enforces the Apple-Music-in-browser
    # short-circuit (the predicate is installed by bobert_companion at boot),
    # so we don't duplicate that guard here — just call through.
    app, _err = itunes_bridge.get_client(wait_for_ready=False, timeout=2.0)
    if app is None:
        return None
    try:
        # PlayerState: 0=stopped, 1=playing, 2=paused (per iTunes COM SDK).
        # We log paused tracks too — the user is still "on" that song.
        ps = getattr(app, "PlayerState", 1)
        if ps == 0:
            return None
        t = app.CurrentTrack
        if t is None:
            return None
        return {
            "artist": getattr(t, "Artist", "") or "",
            "title":  getattr(t, "Name",   "") or "",
            "album":  getattr(t, "Album",  "") or "",
            "genre":  getattr(t, "Genre",  "") or "",
            "source": "itunes",
        }
    except Exception:
        return None


# Apple Music / Spotify web window titles. Apple Music's web player puts the
# title in the form "Song Title — Artist Name – Apple Music" (em dash + en
# dash). Older builds use a hyphen. Spotify uses " · " between title and
# artist on its Web Player.
_APPLE_TITLE_PATTERNS = (
    re.compile(r"^(?P<title>.+?)\s*[—–-]\s*(?P<artist>.+?)\s*[—–-]\s*Apple Music\b", re.I),
    re.compile(r"^(?P<title>.+?)\s*[—–-]\s*(?P<artist>.+?)\s*\|\s*Apple Music\b", re.I),
)
_SPOTIFY_TITLE_PATTERNS = (
    # Spotify's now-playing window title is always `Track · Artist - Spotify`
    # (middle-dot between track and artist). A leading "Spotify - …" title
    # is the idle/library view (e.g. "Spotify - Liked Songs"), never a track,
    # so we deliberately don't try to parse it.
    re.compile(r"^(?P<title>.+?)\s*[·∙•]\s*(?P<artist>.+?)\s*[-–—]\s*Spotify\b", re.I),
)


def _sample_window_title() -> dict | None:
    """Try to read the now-playing track from a Chrome/Edge tab title that
    matches Apple Music's or Spotify's web-player title format."""
    try:
        import pygetwindow as gw
    except ImportError:
        return None
    try:
        windows = gw.getAllWindows()
    except Exception:
        return None
    for w in windows:
        title = (getattr(w, "title", "") or "").strip()
        if not title:
            continue
        # Apple Music web — most-specific match wins; check before "spotify".
        if "apple music" in title.lower():
            # Keep bobert_companion's routing cache warm so play/pause/next/
            # previous never accidentally route to iTunes COM while the user's
            # actual player is the browser. The cache also covers the case
            # where the Apple Music tab is in the background of a Chrome
            # window (Win32 GetWindowText only sees the foreground tab).
            try:
                bc = sys.modules.get("__main__") or sys.modules.get("bobert_companion")
                if bc is not None and hasattr(bc, "_note_apple_music_seen"):
                    bc._note_apple_music_seen()
            except Exception:
                pass
            for pat in _APPLE_TITLE_PATTERNS:
                m = pat.match(title)
                if m:
                    return {
                        "artist": m.group("artist").strip(),
                        "title":  m.group("title").strip(),
                        "album":  "",
                        "genre":  "",
                        "source": "web_apple",
                    }
        if "spotify" in title.lower():
            for pat in _SPOTIFY_TITLE_PATTERNS:
                m = pat.match(title)
                if m:
                    return {
                        "artist": m.group("artist").strip(),
                        "title":  m.group("title").strip(),
                        "album":  "",
                        "genre":  "",
                        "source": "web_spotify",
                    }
    return None


def _sample_now_playing() -> dict | None:
    """Try every source, return the first hit or None."""
    for fn in (_sample_itunes, _sample_window_title):
        try:
            r = fn()
        except Exception:
            r = None
        if r and r.get("artist") and r.get("title"):
            return r
    return None


# ─── listen / skip logging ────────────────────────────────────────────────

def _log_listen(prev: dict, listened_secs: float) -> None:
    """Append a listen event when a track changes. `prev` is the track that
    JUST stopped being current; `listened_secs` is how long it was observed."""
    now = time.time()
    lt = time.localtime(now)
    entry = {
        "ts":       now,
        "iso":      time.strftime("%Y-%m-%dT%H:%M:%S", lt),
        "date":     time.strftime("%Y-%m-%d", lt),
        "day":      time.strftime("%A", lt).lower(),
        "hour":     lt.tm_hour,
        "part":     _time_of_day(lt.tm_hour),
        "artist":   prev.get("artist", ""),
        "title":    prev.get("title", ""),
        "album":    prev.get("album", ""),
        "genre":    prev.get("genre", ""),
        "source":   prev.get("source", ""),
        "secs":     int(listened_secs),
        "complete": listened_secs >= SKIP_THRESHOLD_SECS,
    }
    _append_jsonl(_HISTORY_FILE, entry)
    if not entry["complete"]:
        _append_jsonl(_SKIPS_FILE, entry)
    _maybe_rotate(_HISTORY_FILE)
    _maybe_rotate(_SKIPS_FILE)


def _listen_loop() -> None:
    """Background poll: detect track changes, log durations, mark skips."""
    global _current
    time.sleep(INITIAL_DELAY_SECS)
    while True:
        try:
            sample = _sample_now_playing()
            now = time.time()
            with _listen_lock:
                cur = _current
                if sample is None:
                    # Nothing playing right now. If we had a track in flight,
                    # finalise it (treat as listened up until the last sample).
                    if cur is not None:
                        elapsed = now - cur["since"]
                        _log_listen(cur, elapsed)
                        _current = None
                else:
                    key = _track_key(sample["artist"], sample["title"])
                    if cur is None or cur.get("key") != key:
                        if cur is not None:
                            elapsed = now - cur["since"]
                            _log_listen(cur, elapsed)
                        sample["key"]    = key
                        sample["since"]  = now
                        sample["logged"] = False
                        _current = sample
            time.sleep(POLL_SECONDS)
        except Exception:
            logging.exception("[apple_music_intel] listen_loop iteration failed")
            time.sleep(POLL_SECONDS)


# ─── session skip memory ──────────────────────────────────────────────────

def _load_session() -> dict:
    """Per-process skipped-artist set. Persisted to disk so a JARVIS upgrade
    mid-session doesn't lose the skips, but cleared when the file's
    `session_start` is older than 6 hours (rough session boundary)."""
    if not os.path.exists(_SESSION_FILE):
        return {"session_start": time.time(), "skipped_keys": []}
    try:
        with open(_SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"session_start": time.time(), "skipped_keys": []}
        # Expire stale sessions
        start = float(data.get("session_start", 0) or 0)
        if not start or (time.time() - start) > 6 * 3600:
            return {"session_start": time.time(), "skipped_keys": []}
        return data
    except Exception:
        return {"session_start": time.time(), "skipped_keys": []}


def _save_session(state: dict) -> None:
    # Cap the skip list — guards against pathological "skip everything" runs.
    keys = list(state.get("skipped_keys", []))
    if len(keys) > SESSION_SKIP_CAP:
        keys = keys[-SESSION_SKIP_CAP:]
    state["skipped_keys"] = keys
    try:
        _atomic_write_json(_SESSION_FILE, state)
    except Exception as e:
        print(f"  [apple_music_intel] session save failed: {e}")


def _session_record_skip(artist: str, title: str) -> None:
    state = _load_session()
    key = _track_key(artist, title)
    if key and key not in state.get("skipped_keys", []):
        state.setdefault("skipped_keys", []).append(key)
        _save_session(state)


def _session_skipped_keys() -> set[str]:
    state = _load_session()
    return set(state.get("skipped_keys", []) or [])


# ─── taste aggregation ────────────────────────────────────────────────────

def aggregate() -> dict:
    """Rebuild the taste snapshot from the listen log. Cheap — single pass."""
    events = _read_jsonl(_HISTORY_FILE)
    snapshot: dict = {
        "generated_at":  time.time(),
        "events":        len(events),
        "by_slot":       {},     # "day|part" -> { artist: count, ... }
        "by_artist":     {},     # artist -> { count, last_played_iso, skips }
        "skip_rate_by_genre": {},
        "skip_rate_overall": 0.0,
    }
    if not events:
        _atomic_write_json(_TASTE_FILE, snapshot)
        return snapshot

    by_slot: dict[str, Counter] = defaultdict(Counter)
    by_artist_count: Counter = Counter()
    by_artist_last: dict[str, str] = {}
    by_artist_skips: Counter = Counter()
    by_genre_total: Counter = Counter()
    by_genre_skips: Counter = Counter()

    total_complete = 0
    total_skips = 0

    for e in events:
        artist = (e.get("artist") or "").strip()
        if not artist:
            continue
        day  = (e.get("day") or "").lower()
        part = (e.get("part") or "").lower()
        if day in _DAYS and part in {"morning", "afternoon", "evening", "night"}:
            by_slot[_slot_key(day, part)][artist] += 1

        by_artist_count[artist] += 1
        iso = e.get("iso") or ""
        if iso and iso > by_artist_last.get(artist, ""):
            by_artist_last[artist] = iso

        complete = bool(e.get("complete"))
        if complete:
            total_complete += 1
        else:
            total_skips += 1
            by_artist_skips[artist] += 1

        genre = (e.get("genre") or "").strip()
        if genre:
            by_genre_total[genre] += 1
            if not complete:
                by_genre_skips[genre] += 1

    snapshot["by_slot"] = {
        slot: dict(counter.most_common(20))
        for slot, counter in by_slot.items()
    }
    snapshot["by_artist"] = {
        artist: {
            "count":           by_artist_count[artist],
            "last_played_iso": by_artist_last.get(artist, ""),
            "skips":           by_artist_skips.get(artist, 0),
        }
        for artist in by_artist_count
    }
    skip_rate_by_genre: dict[str, float] = {}
    for genre, total in by_genre_total.items():
        if total < 3:
            continue   # too few samples to be meaningful
        skip_rate_by_genre[genre] = round(by_genre_skips.get(genre, 0) / total, 3)
    snapshot["skip_rate_by_genre"] = dict(
        sorted(skip_rate_by_genre.items(), key=lambda kv: kv[1], reverse=True)
    )
    total_obs = total_complete + total_skips
    snapshot["skip_rate_overall"] = (
        round(total_skips / total_obs, 3) if total_obs else 0.0
    )

    _atomic_write_json(_TASTE_FILE, snapshot)
    return snapshot


def _load_taste() -> dict:
    if not os.path.exists(_TASTE_FILE):
        return {}
    try:
        with open(_TASTE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _aggregator_loop() -> None:
    time.sleep(INITIAL_DELAY_SECS + 10)
    while True:
        try:
            aggregate()
            time.sleep(AGGREGATE_INTERVAL)
        except Exception:
            logging.exception("  [apple_music_intel] aggregator error")
            time.sleep(AGGREGATE_INTERVAL)


# ─── vibe slot parsing ────────────────────────────────────────────────────

def _parse_vibe_slot(arg: str) -> tuple[str, str]:
    """Return (day, part) for a vibe argument. Empty or 'now'/'current'/'today'
    resolves from the current local time."""
    tokens = re.split(r"[\s,_\-]+", (arg or "").lower().strip())
    tokens = [t for t in tokens if t]

    day: str | None = None
    part: str | None = None

    if not tokens or tokens[0] in {"now", "current", "today"}:
        lt = time.localtime()
        return (time.strftime("%A", lt).lower(), _time_of_day(lt.tm_hour))

    for tok in tokens:
        if tok in _DAYS:
            day = tok
        elif tok in _DAY_ALIASES:
            day = _DAY_ALIASES[tok]
        elif tok in _PARTS:
            part = _PARTS[tok]

    if day is None:
        lt = time.localtime()
        day = time.strftime("%A", lt).lower()
    if part is None:
        lt = time.localtime()
        part = _time_of_day(lt.tm_hour)
    return (day, part)


def _top_artist_for_slot(day: str, part: str) -> tuple[str, int] | None:
    snap = _load_taste()
    by_slot = snap.get("by_slot") or {}
    counts = by_slot.get(_slot_key(day, part)) or {}
    if not counts:
        return None
    # Pick the highest-count artist that is NOT in the session skip set.
    skipped = _session_skipped_keys()
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    for artist, cnt in ranked:
        if cnt < MIN_LISTENS_FOR_VIBE:
            continue
        # Skip-set keys are "artist|title" — if every listen of this artist
        # in this slot is from the skip list, skip the artist. Cheap proxy:
        # if the artist's name appears anywhere in a skipped key, skip them.
        a_norm = _norm(artist)
        if any(k.startswith(a_norm + "|") for k in skipped):
            continue
        return (artist, cnt)
    return None


# ─── action handlers ──────────────────────────────────────────────────────

def _bobert():
    """Return the main bobert_companion module — still used for non-iTunes
    helpers (_play_music_core, _act_apple_music, _note_apple_music_seen).
    iTunes COM itself now goes through audio.itunes_bridge directly."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _act_play_unheard(arg: str = "") -> str:
    """Play an iTunes library track that hasn't been played in MIN_DAYS days.
    Arg: optional integer days override."""
    try:
        min_days = max(1, int((arg or "").strip()))
    except (ValueError, TypeError):
        min_days = UNHEARD_MIN_DAYS

    # `play_unheard` is an explicit iTunes-library-only intent — Apple Music
    # streaming has no equivalent of "track I haven't played in N days", so
    # pass force=True to bypass the Apple-Music-active short-circuit inside
    # the bridge. Still gated by ITUNES_AUTO_LAUNCH for the "iTunes isn't
    # running" path, so this won't spawn iTunes from cold.
    app, err = itunes_bridge.get_client(force=True)
    if app is None:
        return err or "iTunes not reachable, sir."

    cutoff_ts = time.time() - min_days * 86400
    skipped = _session_skipped_keys()

    try:
        import pythoncom  # noqa: F401  (already coinit'd inside itunes_bridge.get_client)
        lib = app.LibraryPlaylist
        # Walk the library top-down but cap iteration so very large libs
        # don't stall the loop. iTunes COM exposes Tracks as a 1-based
        # IITTrackCollection — we sample sequentially and keep candidates.
        total = lib.Tracks.Count
        if not total:
            return "iTunes library is empty, sir."
        # Use a strided sample so we don't always start at the same place.
        # Caps at ~1500 tracks scanned regardless of library size.
        max_scan = min(total, 1500)
        step = max(1, total // max_scan)
        candidates: list[tuple[object, str, str, float]] = []
        i = 1
        scanned = 0
        while i <= total and scanned < max_scan:
            try:
                t = lib.Tracks.Item(i)
            except Exception:
                i += step; scanned += 1; continue
            try:
                name   = getattr(t, "Name",   "") or ""
                artist = getattr(t, "Artist", "") or ""
                if not name or not artist:
                    i += step; scanned += 1; continue
                pd = getattr(t, "PlayedDate", None)
                # PlayedDate is a pywintypes datetime when present, or a
                # sentinel "1899-12-30" when never played. Treat both
                # "never played" AND "played before cutoff" as eligible.
                played_ts = 0.0
                if pd is not None:
                    try:
                        # pywintypes datetime supports .timestamp() in pywin32 ≥ 224
                        played_ts = float(pd.timestamp())  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            played_ts = float(time.mktime(pd.timetuple()))
                        except Exception:
                            played_ts = 0.0
                # iTunes uses 1899-12-30 (~ -2208988800) as the never-played
                # sentinel — anything before year 1990 we treat as 0.
                if played_ts and played_ts < 631152000:
                    played_ts = 0.0
                if played_ts and played_ts >= cutoff_ts:
                    i += step; scanned += 1; continue
                key = _track_key(artist, name)
                if key in skipped:
                    i += step; scanned += 1; continue
                candidates.append((t, name, artist, played_ts))
            except Exception:
                pass
            i += step
            scanned += 1
        if not candidates:
            return (f"No tracks unheard for {min_days}+ days in the visible "
                    f"slice of your library, sir.")
        # Prefer the longest-unheard candidate (lowest played_ts, 0 = never).
        candidates.sort(key=lambda c: c[3])
        track, name, artist, played_ts = candidates[0]
        track.Play()
        when = ("never before" if not played_ts
                else f"last on {time.strftime('%b %d, %Y', time.localtime(played_ts))}")
        return f"Playing '{name}' by {artist}, sir — {when}."
    except Exception as e:
        return f"unheard search failed: {e}"


def _act_play_vibe(arg: str = "") -> str:
    """Play the dominant artist for a given (day, part-of-day) slot."""
    day, part = _parse_vibe_slot(arg)
    top = _top_artist_for_slot(day, part)
    if top is None:
        # Re-aggregate once in case taste file is stale, then retry.
        try:
            aggregate()
        except Exception:
            pass
        top = _top_artist_for_slot(day, part)
    if top is None:
        return (f"I don't have a strong {day} {part} pattern yet, sir — "
                f"need at least {MIN_LISTENS_FOR_VIBE} prior listens in that "
                f"slot before I'd dare presume.")
    artist, cnt = top

    # Prefer iTunes COM (gives us a real track), fall back to apple_music
    # web auto-play if iTunes isn't usable.
    bc = _bobert()
    if bc is not None and hasattr(bc, "_play_music_core"):
        try:
            ok, res = bc._play_music_core(f"artist:{artist}")
            if ok:
                return f"Vibing your usual {day} {part}, sir — {res}"
        except Exception:
            pass
    if bc is not None and hasattr(bc, "_act_apple_music"):
        try:
            res = bc._act_apple_music(artist)
            return f"Vibing your usual {day} {part}, sir — {res}"
        except Exception:
            pass
    return f"Top {day} {part} artist is {artist} ({cnt} prior listens), sir — but I can't reach a player to queue it."


def _act_skip_track(_: str = "") -> str:
    """Record the current track as skipped-this-session, then skip it."""
    with _listen_lock:
        cur = dict(_current) if _current else None

    artist = (cur or {}).get("artist", "") if cur else ""
    title  = (cur or {}).get("title", "")  if cur else ""

    # If we don't know what's playing yet, still skip — best-effort.
    if cur is None:
        bc = _bobert()
        sampled = _sample_now_playing()
        if sampled:
            artist = sampled.get("artist", "") or artist
            title  = sampled.get("title", "")  or title

    if artist and title:
        _session_record_skip(artist, title)
        # Also write a synthetic skip event so the taste model picks it up
        # immediately instead of waiting until the track changes naturally.
        now = time.time()
        lt = time.localtime(now)
        entry = {
            "ts":       now,
            "iso":      time.strftime("%Y-%m-%dT%H:%M:%S", lt),
            "date":     time.strftime("%Y-%m-%d", lt),
            "day":      time.strftime("%A", lt).lower(),
            "hour":     lt.tm_hour,
            "part":     _time_of_day(lt.tm_hour),
            "artist":   artist,
            "title":    title,
            "album":    (cur or {}).get("album", ""),
            "genre":    (cur or {}).get("genre", ""),
            "source":   (cur or {}).get("source", "user_skip"),
            "secs":     0,
            "complete": False,
            "user_skip": True,
        }
        _append_jsonl(_HISTORY_FILE, entry)
        _append_jsonl(_SKIPS_FILE, entry)

    # Issue the actual skip — try iTunes COM first, fall back to media key.
    bc = _bobert()
    skipped_via = None
    if bc is not None and hasattr(bc, "_act_next_song") and (cur or {}).get("source") == "itunes":
        try:
            bc._act_next_song("")
            skipped_via = "iTunes"
        except Exception:
            pass
    if skipped_via is None and bc is not None and hasattr(bc, "_act_media_next"):
        try:
            bc._act_media_next("")
            skipped_via = "media key"
        except Exception:
            pass

    track_note = f"'{title}' by {artist}" if (artist and title) else "the current track"
    where = f" (via {skipped_via})" if skipped_via else ""
    return f"Noted, sir — I'll set {track_note} aside for the rest of the session{where}."


def _act_music_history(_: str = "") -> str:
    """Read back the last 5 listens with timestamps."""
    events = _read_jsonl(_HISTORY_FILE)
    if not events:
        return "No listening history yet, sir."
    last = events[-5:]
    parts = []
    for e in last:
        when = (e.get("iso") or "").replace("T", " ")
        secs = e.get("secs", 0)
        tag  = "" if e.get("complete") else " (skipped)"
        parts.append(f"{when} — '{e.get('title','?')}' by "
                     f"{e.get('artist','?')} [{secs}s{tag}]")
    return "Recent listens, sir: " + "; ".join(parts)


def _act_music_taste(_: str = "") -> str:
    """Short readback of the dominant taste pattern + skip rate by genre."""
    snap = _load_taste()
    if not snap or not snap.get("by_artist"):
        return "No taste data yet, sir — I haven't observed enough listens."
    by_artist = snap.get("by_artist") or {}
    top_artists = sorted(by_artist.items(),
                         key=lambda kv: kv[1].get("count", 0), reverse=True)[:3]
    artist_str = ", ".join(f"{a} ({d.get('count',0)})" for a, d in top_artists)

    lt = time.localtime()
    day  = time.strftime("%A", lt).lower()
    part = _time_of_day(lt.tm_hour)
    slot_top = _top_artist_for_slot(day, part)
    slot_str = (f"{slot_top[0]} dominates {day} {part}s ({slot_top[1]} listens)"
                if slot_top else f"no strong {day} {part} pattern yet")

    skip_rate = snap.get("skip_rate_by_genre") or {}
    if skip_rate:
        worst = next(iter(skip_rate.items()))
        skip_str = f"highest skip rate: {worst[0]} at {int(worst[1]*100)}%"
    else:
        overall = snap.get("skip_rate_overall", 0.0) or 0.0
        skip_str = f"overall skip rate {int(overall*100)}%"

    return (f"Top artists, sir: {artist_str or 'none yet'}. "
            f"{slot_str.capitalize()}. {skip_str.capitalize()}.")


def _act_music_aggregate(_: str = "") -> str:
    snap = aggregate()
    n_events = snap.get("events", 0)
    n_artists = len(snap.get("by_artist") or {})
    n_slots = len(snap.get("by_slot") or {})
    return (f"Aggregated {n_events} listen events across {n_artists} artists "
            f"and {n_slots} day/time slots, sir.")


# ─── registration ─────────────────────────────────────────────────────────

def register(actions):
    actions["play_unheard"]    = _act_play_unheard
    actions["play_vibe"]       = _act_play_vibe
    actions["skip_track"]      = _act_skip_track
    actions["music_history"]   = _act_music_history
    actions["music_taste"]     = _act_music_taste
    actions["music_aggregate"] = _act_music_aggregate

    _ensure_data_dir()

    # Fresh JARVIS process = new session. Clear stale session skips so a
    # restart doesn't suppress everything the user previously skipped today.
    try:
        _save_session({"session_start": time.time(), "skipped_keys": []})
    except Exception:
        pass

    # Guard against duplicate loops on skill reload: load_skills() re-execs
    # this module so _aggregator_started resets to False — a module flag can't
    # see a prior load's still-running thread. Check live OS threads by name.
    _alive = {th.name for th in threading.enumerate() if th.is_alive()}
    if "am-intel-listen" not in _alive:
        threading.Thread(target=_listen_loop, daemon=True,
                         name="am-intel-listen").start()
    if "am-intel-aggregate" not in _alive:
        threading.Thread(target=_aggregator_loop, daemon=True,
                         name="am-intel-aggregate").start()
        _aggregator_started[0] = True
    print(f"  [apple_music_intel] tracking listens every {POLL_SECONDS}s; "
          f"taste snapshot every {AGGREGATE_INTERVAL//60}m")


# ─── offline smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate synthetic listen events and exercise aggregate + slot lookup.
    print("Running apple_music_intel smoke test…")
    tmpdir = tempfile.mkdtemp(prefix="amintel_")
    globals()["_DATA_DIR"]     = tmpdir
    globals()["_HISTORY_FILE"] = os.path.join(tmpdir, "history.jsonl")
    globals()["_SKIPS_FILE"]   = os.path.join(tmpdir, "skips.jsonl")
    globals()["_SESSION_FILE"] = os.path.join(tmpdir, "session.json")
    globals()["_TASTE_FILE"]   = os.path.join(tmpdir, "taste.json")
    base = time.time() - 30 * 86400
    fake = []
    for day_off in range(30):
        ts = base + day_off * 86400
        lt = time.localtime(ts)
        wd_name = time.strftime("%A", lt).lower()
        # On Fridays at 21:00, play Michael Jackson 3 times
        if wd_name == "friday":
            for k in range(3):
                fake.append({
                    "ts": ts + k * 200, "iso": time.strftime("%Y-%m-%dT21:%M:00", lt),
                    "date": time.strftime("%Y-%m-%d", lt), "day": wd_name,
                    "hour": 21, "part": "evening",
                    "artist": "Michael Jackson", "title": f"Track {k}",
                    "album": "X", "genre": "Pop", "source": "itunes",
                    "secs": 200, "complete": True,
                })
        # On Sunday mornings, play Bon Iver
        if wd_name == "sunday":
            fake.append({
                "ts": ts, "iso": time.strftime("%Y-%m-%dT08:30:00", lt),
                "date": time.strftime("%Y-%m-%d", lt), "day": wd_name,
                "hour": 8, "part": "morning",
                "artist": "Bon Iver", "title": "Holocene",
                "album": "Bon Iver", "genre": "Indie", "source": "itunes",
                "secs": 240, "complete": True,
            })
        # Some skipped country tracks to test skip_rate_by_genre
        if day_off % 2 == 0:
            fake.append({
                "ts": ts, "iso": time.strftime("%Y-%m-%dT15:00:00", lt),
                "date": time.strftime("%Y-%m-%d", lt), "day": wd_name,
                "hour": 15, "part": "afternoon",
                "artist": "Some Country Act", "title": "Nope",
                "album": "Y", "genre": "Country", "source": "itunes",
                "secs": 8, "complete": False,
            })
    for e in fake:
        _append_jsonl(_HISTORY_FILE, e)
    snap = aggregate()
    print(f"  events: {snap['events']}, artists: {len(snap['by_artist'])}, "
          f"slots: {len(snap['by_slot'])}")
    print(f"  friday/evening top: {_top_artist_for_slot('friday', 'evening')}")
    print(f"  sunday/morning top: {_top_artist_for_slot('sunday', 'morning')}")
    print(f"  skip_rate_by_genre: {snap['skip_rate_by_genre']}")
    print(f"  parse 'friday night': {_parse_vibe_slot('friday night')}")
    print(f"  parse 'now': {_parse_vibe_slot('now')}")
    print(f"  parse '': {_parse_vibe_slot('')}")
    # Cleanup
    for fn in ("history.jsonl", "skips.jsonl", "session.json", "taste.json"):
        try: os.unlink(os.path.join(tmpdir, fn))
        except Exception: pass
    try: os.rmdir(tmpdir)
    except Exception: pass
    print("Smoke test complete.")
