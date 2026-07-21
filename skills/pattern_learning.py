"""
Pattern learning — nightly aggregator that turns the action-event log into
predictions about the user's habits, then surfaces those predictions through
the anticipation engine.

This is a companion to memory.py (which mines voice-utterance patterns).
This module focuses on ACTION events — what JARVIS actually did — and
produces two prediction shapes the spec asks for:

  • Broad-window  "the user plays music between 9-11am on weekdays 78% of the time"
  • Precise-clock "the user checks Teams at 09:15 ± 5 min daily"

Storage:
  data/usage_patterns.jsonl              — append-only event log (legacy /
                                            backward-compat for daily_recap.py)
  data/usage_patterns.sqlite3            — same events as SQLite for fast
                                            day-of-week / hour-window queries
                                            (events + weekly_summaries tables).
  data/usage_patterns_aggregated.json    — nightly aggregator snapshot
  data/usage_patterns_offer_state.json   — once-per-day per-pattern throttle

Logging entry point:
  log_event(action_name, arg)            — called from
                                            bobert_companion.record_session_action.
                                            Writes BOTH JSONL and SQLite so
                                            existing consumers stay working
                                            while new code can query SQL.

Anticipation-engine entry point:
  maybe_pattern_offer_v2()               — memory.maybe_pattern_offer reads this
                                            first when the skill is loaded

Weekly digest:
  compute_weekly_digest()                — cluster events from the last 7 days
                                            by (dow, hour-window, action) and
                                            cache the result in the SQLite
                                            weekly_summaries table.
  load_latest_weekly_digest()            — read the most recent cached digest.

Actions registered:
  pattern_predictions     — speak the top 5 detected predictions
  pattern_offer_now       — force the next eligible offer (bypasses throttle)
  pattern_aggregate       — rerun the nightly aggregation right now
  pattern_stats           — short status report
  weekly_digest           — recompute + speak the latest weekly habit summary
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
import statistics
import tempfile
import threading
import time
from collections import Counter, defaultdict

_HERE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_HERE, "data")
_LOG_FILE    = os.path.join(_DATA_DIR, "usage_patterns.jsonl")
_DB_FILE     = os.path.join(_DATA_DIR, "usage_patterns.sqlite3")
_AGG_FILE    = os.path.join(_DATA_DIR, "usage_patterns_aggregated.json")
_STATE_FILE  = os.path.join(_DATA_DIR, "usage_patterns_offer_state.json")

# Logging
MAX_LOG_ENTRIES   = 20000          # rotate the jsonl down to this when exceeded
DEBOUNCE_SECONDS  = 1.0            # collapse identical back-to-back events
_log_lock         = threading.Lock()
_last_event       = {"key": "", "ts": 0.0}
_writes_since_rotate = [0]

# Aggregation thresholds
MIN_DAYS_OBSERVED        = 7       # need a week of data before predicting
MIN_OCCURRENCES_BROAD    = 3
MIN_RATIO_BROAD          = 0.5     # action must fire in ≥50% of bucket days
BROAD_WINDOW_HOURS       = 2       # rolling 2-hour windows: 0-2, 1-3, … 22-24

MIN_OCCURRENCES_PRECISE  = 4
MIN_RATIO_PRECISE        = 0.6
STDDEV_THRESHOLD_MIN     = 12.0    # tighter than this and we call it precise

# Background aggregator
INITIAL_DELAY_SECONDS    = 30
LOOP_INTERVAL_SECONDS    = 600     # 10 min
NIGHTLY_HOUR             = 3       # local-clock hour to run nightly
MAX_AGGREGATE_AGE_SECS   = 12 * 3600

# Weekly digest
WEEKLY_DIGEST_DOW        = 0       # 0=Mon … 6=Sun. Run weekly on this DOW.
WEEKLY_DIGEST_HOUR       = 9       # local-clock hour to run the weekly digest
WEEKLY_LOOKBACK_DAYS     = 28      # consider up to four weeks of history
WEEKLY_MIN_OCCURRENCES   = 3       # at least 3 hits to form a cluster
WEEKLY_MIN_WEEKS         = 2       # span at least 2 distinct weeks
WEEKLY_HOUR_WINDOW       = 2       # bucket size in hours (e.g. 8-10pm)
WEEKLY_MAX_CARDS         = 5       # cap returned clusters
WEEKLY_AGE_MAX_SECS      = 8 * 86400  # consider a cached digest stale after ~8d

# Friendly labels for the predictions readback. Anything not listed falls back
# to a generic verb derived from the action name.
_ACTION_VERB = {
    "play_music":      ("plays music",            "Shall I queue your usual mix, sir?"),
    "resume_music":    ("resumes music",          "Shall I resume your usual playlist, sir?"),
    "apple_music":     ("opens Apple Music",      "Apple Music ready when you are, sir."),
    "spotify":         ("opens Spotify",          "Shall I queue Spotify, sir?"),
    "youtube":         ("opens YouTube",          "Shall I bring up YouTube, sir?"),
    "youtube_play":    ("plays YouTube",          "Shall I queue your usual YouTube watch, sir?"),
    "netflix":         ("opens Netflix",          "Shall I queue something on Netflix, sir?"),
    "prime_video":     ("opens Prime Video",      "Prime Video, sir?"),
    "disney_plus":     ("opens Disney+",          "Disney+, sir?"),
    "hulu":            ("opens Hulu",             "Hulu, sir?"),
    "max":             ("opens Max",              "Max, sir?"),
    "check_teams":     ("checks Teams",           "You usually check Teams around now, sir — shall I take a look?"),
    "check_credits":   ("checks credits",         "Shall I check the credits balance, sir?"),
    "check_system":    ("runs a systems check",   "Shall I run a systems check, sir?"),
    "check_print":     ("checks the printer",     "Shall I check the printer, sir?"),
    "morning_briefing":("runs the morning briefing", "Shall I deliver the morning briefing, sir?"),
    "evening_briefing":("runs the evening briefing", "Shall I deliver the evening briefing, sir?"),
    "open_url":        ("opens a browser tab",    "Shall I bring up your usual page, sir?"),
    "launch_app":      ("launches an app",        "Shall I launch your usual app, sir?"),
    "queue_task":      ("queues a task",          "Anything to add to the queue, sir?"),
    "show_tasks":      ("reviews the task queue", "Shall I run through the task queue, sir?"),
    "upgrade":         ("kicks off an upgrade",   "Shall I run an upgrade, sir?"),
    "start_overnight_upgrade": ("starts the overnight upgrade engine",
                                "Shall I start the overnight upgrader, sir?"),
}


# ─── data dir / file helpers ─────────────────────────────────────────────

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


# ─── SQLite (parallel store for the weekly digest queries) ───────────────
#
# JSONL stays the source of truth for backward compatibility (daily_recap.py
# still reads it directly). SQLite is a parallel mirror so weekly grouping by
# (dow, hour-window, action) is one SQL aggregate instead of an O(n) Python
# scan. Both writes happen inside log_event(); the SQLite write is best-effort
# so a missing or locked DB never blocks log ingestion.

_db_lock = threading.Lock()
_db_initialized = [False]
_backfill_lock = threading.Lock()
_backfill_done = threading.Event()

_SCHEMA_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL    NOT NULL,
    iso     TEXT    NOT NULL,
    date    TEXT    NOT NULL,
    dow     INTEGER NOT NULL,   -- 0=Mon … 6=Sun
    hour    INTEGER NOT NULL,
    minute  INTEGER NOT NULL,
    action  TEXT    NOT NULL,
    arg     TEXT    NOT NULL DEFAULT ''
);
"""
_SCHEMA_EVENTS_IDX_TS     = "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);"
_SCHEMA_EVENTS_IDX_ACTION = "CREATE INDEX IF NOT EXISTS idx_events_action ON events(action);"
_SCHEMA_WEEKLY = """
CREATE TABLE IF NOT EXISTS weekly_summaries (
    week_start    TEXT PRIMARY KEY,    -- ISO date of Monday for the snapshot
    computed_at   REAL NOT NULL,
    cluster_data  TEXT NOT NULL        -- JSON list[dict] of clusters
);
"""


def _connect_db() -> sqlite3.Connection | None:
    """Return a short-lived DB connection. Lazily creates the schema. Returns
    None if sqlite3 can't open the file (e.g. data dir not writable) — callers
    must tolerate a missing DB so JSONL ingestion never breaks."""
    _ensure_data_dir()
    try:
        conn = sqlite3.connect(_DB_FILE, timeout=2.0, isolation_level=None)
    except Exception as e:
        print(f"  [pattern_learning] sqlite connect failed: {e}")
        return None
    if not _db_initialized[0]:
        with _db_lock:
            if not _db_initialized[0]:
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute(_SCHEMA_EVENTS)
                    conn.execute(_SCHEMA_EVENTS_IDX_TS)
                    conn.execute(_SCHEMA_EVENTS_IDX_ACTION)
                    conn.execute(_SCHEMA_WEEKLY)
                    _db_initialized[0] = True
                except Exception as e:
                    print(f"  [pattern_learning] sqlite schema init failed: {e}")
                    try: conn.close()
                    except Exception: pass  # pragma: no cover - defensive close after schema-init failure
                    return None
    return conn


def _backfill_sqlite_from_jsonl_if_empty() -> int:
    """One-shot migration helper. If the events table is empty but the JSONL
    has rows, replay the JSONL into SQLite so the weekly digest has data on
    the first boot after this skill upgrade. Returns the number of rows
    backfilled (0 when nothing to do).

    Gated by a dedicated once-flag (_backfill_done) so concurrent callers
    can't both pass the COUNT == 0 check and double-insert the JSONL. We
    can't reuse _db_lock here because _connect_db() acquires it during
    schema init (non-reentrant); the dedicated lock serializes only the
    backfill itself."""
    if _backfill_done.is_set():
        return 0
    with _backfill_lock:
        if _backfill_done.is_set():
            return 0
        conn = _connect_db()
        if conn is None:
            return 0
        try:
            row = conn.execute("SELECT COUNT(1) FROM events").fetchone()
            if row and row[0] > 0:
                _backfill_done.set()
                return 0
            events = _load_events()
            if not events:
                _backfill_done.set()
                return 0
            inserted = 0
            conn.execute("BEGIN")
            try:
                for e in events:
                    try:
                        conn.execute(
                            "INSERT INTO events(ts, iso, date, dow, hour, minute, action, arg) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                float(e.get("ts") or 0.0),
                                str(e.get("iso") or ""),
                                str(e.get("date") or ""),
                                int(e.get("wd") if isinstance(e.get("wd"), int) else 0),
                                int(e.get("hour") if isinstance(e.get("hour"), int) else 0),
                                int(e.get("min") if isinstance(e.get("min"), int) else 0),
                                str(e.get("action") or ""),
                                str(e.get("arg") or ""),
                            ),
                        )
                        inserted += 1
                    except Exception:
                        continue
                conn.execute("COMMIT")
            except Exception:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                raise
            _backfill_done.set()
            if inserted:
                print(f"  [pattern_learning] sqlite backfill: {inserted} events from JSONL")
            return inserted
        except Exception as e:
            print(f"  [pattern_learning] sqlite backfill failed: {e}")
            return 0
        finally:
            try: conn.close()
            except Exception: pass  # pragma: no cover - defensive close in backfill finally


def _prune_sqlite_events(max_rows: int = MAX_LOG_ENTRIES) -> None:
    """Keep the events table from growing without bound. Mirrors the JSONL
    rotation policy — same MAX_LOG_ENTRIES cap, oldest rows by ts."""
    conn = _connect_db()
    if conn is None:
        return
    try:
        with _db_lock:
            row = conn.execute("SELECT COUNT(1) FROM events").fetchone()
            n = int(row[0]) if row else 0
            if n <= max_rows:
                return
            excess = n - max_rows
            conn.execute(
                "DELETE FROM events WHERE id IN ("
                "  SELECT id FROM events ORDER BY ts ASC LIMIT ?"
                ")",
                (excess,),
            )
    except Exception as e:
        print(f"  [pattern_learning] sqlite prune failed: {e}")
    finally:
        try: conn.close()
        except Exception: pass  # pragma: no cover - defensive close in prune finally


# ─── logging ─────────────────────────────────────────────────────────────

def log_event(action_name: str, arg: str = "") -> None:
    """Append one action-execution event to data/usage_patterns.jsonl.

    Called from bobert_companion.record_session_action for every action that
    actually ran. Cheap (single file append). Debounces identical events
    within DEBOUNCE_SECONDS so a follow-up loop that re-emits the same action
    twice in a row doesn't pollute the log."""
    if not action_name:
        return
    arg = (arg or "").strip()
    key = f"{action_name}|{arg[:80].lower()}"
    now = time.time()
    with _log_lock:
        if key == _last_event["key"] and (now - _last_event["ts"]) < DEBOUNCE_SECONDS:
            return
        _last_event["key"] = key
        _last_event["ts"]  = now
        lt = time.localtime(now)
        entry = {
            "ts":   now,
            "iso":  time.strftime("%Y-%m-%dT%H:%M:%S", lt),
            "date": time.strftime("%Y-%m-%d", lt),
            "dow":  time.strftime("%A", lt),
            "wd":   lt.tm_wday,                   # 0=Mon … 6=Sun
            "hour": lt.tm_hour,
            "min":  lt.tm_min,
            "action": action_name,
            "arg":  arg[:120],
        }
        _ensure_data_dir()
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [pattern_learning] log write failed: {e}")
            return
        # Mirror into SQLite. Best-effort: a missing/locked DB must NOT block
        # the primary JSONL log path. Same lock as log_lock is already held.
        try:
            conn = _connect_db()
            if conn is not None:
                try:
                    with _db_lock:
                        conn.execute(
                            "INSERT INTO events(ts, iso, date, dow, hour, minute, action, arg) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (entry["ts"], entry["iso"], entry["date"],
                             entry["wd"], entry["hour"], entry["min"],
                             entry["action"], entry["arg"]),
                        )
                finally:
                    try: conn.close()
                    except Exception: pass  # pragma: no cover - defensive close in log-mirror finally
        except Exception as e:
            print(f"  [pattern_learning] sqlite log mirror failed: {e}")
        _writes_since_rotate[0] += 1
        if _writes_since_rotate[0] >= 200:
            _writes_since_rotate[0] = 0
            _maybe_rotate()
            _prune_sqlite_events()


def _maybe_rotate() -> None:
    """Cap the jsonl at MAX_LOG_ENTRIES — keep most recent N lines."""
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= MAX_LOG_ENTRIES:
            return
        keep = lines[-MAX_LOG_ENTRIES:]
        tmp = _LOG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, _LOG_FILE)
    except Exception:
        pass


def _load_events() -> list[dict]:
    if not os.path.exists(_LOG_FILE):
        return []
    out: list[dict] = []
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
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


# ─── aggregation ─────────────────────────────────────────────────────────

def _bucket_for_weekday(wd: int) -> str:
    return "weekday" if 0 <= wd <= 4 else "weekend"


def _format_hour_window(start: int, end: int) -> str:
    def _fmt(h: int) -> str:
        suffix = "am" if h < 12 else "pm"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return f"{h12}{suffix}"
    return f"{_fmt(start)}-{_fmt(end)}"


def _format_clock(minute_of_day: int) -> str:
    h, m = divmod(int(minute_of_day) % (24 * 60), 60)
    return f"{h:02d}:{m:02d}"


def _verb_for(action: str) -> str:
    if action in _ACTION_VERB:
        return _ACTION_VERB[action][0]
    return f"runs {action.replace('_', ' ')}"


def _offer_for(action: str) -> str:
    if action in _ACTION_VERB:
        return _ACTION_VERB[action][1]
    return f"Shall I {action.replace('_', ' ')}, sir?"


def aggregate() -> dict:
    """Rebuild the predictions snapshot from data/usage_patterns.jsonl and
    persist it to data/usage_patterns_aggregated.json. Returns the new
    snapshot dict."""
    events = _load_events()
    snapshot: dict = {
        "generated_at": time.time(),
        "events":       len(events),
        "days_span":    0.0,
        "broad":        [],
        "precise":      [],
    }
    if len(events) < MIN_OCCURRENCES_BROAD:
        _atomic_write_json(_AGG_FILE, snapshot)
        return snapshot

    timestamps = [e.get("ts") for e in events if isinstance(e.get("ts"), (int, float))]
    if timestamps:
        snapshot["days_span"] = round((max(timestamps) - min(timestamps)) / 86400.0, 2)

    # Distinct dates the user was active in each bucket. Used as denominators
    # so the ratio is "fraction of bucket-days on which this action fired in
    # this window", not "fraction of total bucket-events that fired here".
    dates_by_bucket: dict[str, set[str]] = defaultdict(set)
    for e in events:
        wd = e.get("wd")
        date = e.get("date")
        if not isinstance(wd, int) or not date:
            continue
        dates_by_bucket[_bucket_for_weekday(wd)].add(date)

    # ── Broad-window predictions ──
    # For each (action, bucket, 2-hour window): count distinct dates that
    # action fired in this window, and emit if it dominates the bucket's days.
    by_action_bucket_window: dict[tuple[str, str, int], dict] = defaultdict(
        lambda: {"dates": set(), "args": Counter(), "count": 0}
    )
    for e in events:
        action = e.get("action")
        wd = e.get("wd")
        hour = e.get("hour")
        date = e.get("date")
        arg = (e.get("arg") or "").strip().lower()
        if not action or not date or not isinstance(wd, int) or not isinstance(hour, int):
            continue
        if not (0 <= hour <= 23):
            continue
        bucket = _bucket_for_weekday(wd)
        # Each event contributes to every 2-hour window it falls inside —
        # rolling windows W=H..H+2 for H ∈ 0..22.
        for window_start in range(max(0, hour - BROAD_WINDOW_HOURS + 1),
                                  min(23, hour) + 1):
            window_end = window_start + BROAD_WINDOW_HOURS
            if window_end > 24:
                continue
            slot = by_action_bucket_window[(action, bucket, window_start)]
            slot["dates"].add(date)
            slot["count"] += 1
            if arg:
                slot["args"][arg] += 1

    broad_candidates: list[dict] = []
    for (action, bucket, window_start), data in by_action_bucket_window.items():
        days_with_event = len(data["dates"])
        if days_with_event < MIN_OCCURRENCES_BROAD:
            continue
        bucket_days = len(dates_by_bucket.get(bucket, set()))
        if bucket_days < MIN_DAYS_OBSERVED:
            continue
        ratio = days_with_event / bucket_days
        if ratio < MIN_RATIO_BROAD:
            continue
        window_end = window_start + BROAD_WINDOW_HOURS
        pct = int(round(ratio * 100))
        common_arg = data["args"].most_common(1)[0][0] if data["args"] else ""
        _uname = os.getenv("JARVIS_USER_NAME", "").strip() or "You"
        label = (f"{_uname} {_verb_for(action)} between "
                 f"{_format_hour_window(window_start, window_end)} on "
                 f"{bucket}s {pct}% of the time")
        broad_candidates.append({
            "key":             f"broad|{action}|{bucket}|{window_start}-{window_end}",
            "type":            "broad",
            "action":          action,
            "bucket":          bucket,
            "hour_window":     [window_start, window_end],
            "occurrences":     data["count"],
            "days_with_event": days_with_event,
            "days_observed":   bucket_days,
            "ratio":           round(ratio, 3),
            "common_arg":      common_arg,
            "label":           label,
            "offer":           _offer_for(action),
        })

    # Keep only the strongest window per (action, bucket) so we don't surface
    # 1-3, 2-4, 3-5 all together for the same daily habit.
    best_by_action_bucket: dict[tuple[str, str], dict] = {}
    for c in broad_candidates:
        ab = (c["action"], c["bucket"])
        cur = best_by_action_bucket.get(ab)
        score = c["ratio"] * c["days_with_event"]
        if cur is None or score > (cur["ratio"] * cur["days_with_event"]):
            best_by_action_bucket[ab] = c
    snapshot["broad"] = sorted(
        best_by_action_bucket.values(),
        key=lambda x: (x["ratio"], x["days_with_event"]),
        reverse=True,
    )

    # ── Precise-clock predictions ──
    # Cluster by action — if its event minutes-of-day are tightly grouped
    # across many distinct days, emit a "X at HH:MM ± N min" prediction.
    minutes_by_action: dict[str, list[int]] = defaultdict(list)
    dates_by_action:   dict[str, set[str]]  = defaultdict(set)
    for e in events:
        action = e.get("action")
        hour   = e.get("hour")
        minute = e.get("min")
        date   = e.get("date")
        if not action or not date:
            continue
        if not isinstance(hour, int) or not isinstance(minute, int):
            continue
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            continue
        minutes_by_action[action].append(hour * 60 + minute)
        dates_by_action[action].add(date)

    precise: list[dict] = []
    all_active_days = len({e.get("date") for e in events if e.get("date")})
    for action, minutes in minutes_by_action.items():
        if len(minutes) < MIN_OCCURRENCES_PRECISE:
            continue
        if len(minutes) < 2:  # pragma: no cover - unreachable: MIN_OCCURRENCES_PRECISE>=4 already guarantees len>=4
            continue
        try:
            stddev = statistics.pstdev(minutes)
        except statistics.StatisticsError:  # pragma: no cover - unreachable: pstdev only raises for <1 sample; len>=4 here
            continue
        if stddev > STDDEV_THRESHOLD_MIN:
            continue
        days_with_event = len(dates_by_action[action])
        if days_with_event < MIN_OCCURRENCES_PRECISE:
            continue
        if all_active_days < MIN_DAYS_OBSERVED:
            continue
        ratio = days_with_event / all_active_days
        if ratio < MIN_RATIO_PRECISE:
            continue
        center = int(round(statistics.mean(minutes)))
        tolerance = max(5, int(round(stddev)))
        clock = _format_clock(center)
        cadence = "daily" if ratio >= 0.85 else "most days"
        _uname = os.getenv("JARVIS_USER_NAME", "").strip() or "You"
        label = (f"{_uname} {_verb_for(action)} at {clock} "
                 f"± {tolerance} min {cadence}")
        precise.append({
            "key":             f"precise|{action}|{clock}",
            "type":            "precise",
            "action":          action,
            "center_minute":   center,
            "center_clock":    clock,
            "tolerance_min":   tolerance,
            "occurrences":     len(minutes),
            "days_with_event": days_with_event,
            "days_observed":   all_active_days,
            "ratio":           round(ratio, 3),
            "stddev_min":      round(stddev, 1),
            "label":           label,
            "offer":           _offer_for(action),
        })
    snapshot["precise"] = sorted(
        precise,
        key=lambda x: (x["ratio"], -x["stddev_min"]),
        reverse=True,
    )

    _atomic_write_json(_AGG_FILE, snapshot)
    return snapshot


def _load_aggregated() -> dict:
    if not os.path.exists(_AGG_FILE):
        return {}
    try:
        with open(_AGG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def predictions_for_now(snapshot: dict | None = None) -> list[dict]:
    """Return predictions whose time window matches the current moment.
    Used for proactive offers."""
    if snapshot is None:
        snapshot = _load_aggregated()
    if not snapshot:
        return []
    now = time.localtime()
    wd  = now.tm_wday
    cur_bucket = _bucket_for_weekday(wd)
    cur_minute = now.tm_hour * 60 + now.tm_min

    matches: list[dict] = []
    for p in snapshot.get("broad", []):
        if p.get("bucket") != cur_bucket:
            continue
        win = p.get("hour_window") or []
        if len(win) != 2:
            continue
        if win[0] <= now.tm_hour < win[1]:
            matches.append(p)
    for p in snapshot.get("precise", []):
        center = p.get("center_minute")
        tol    = p.get("tolerance_min", 0)
        if not isinstance(center, int) or not isinstance(tol, int):
            continue
        if abs(cur_minute - center) <= tol:
            matches.append(p)
    # Sort: precise first (more specific), then by ratio
    matches.sort(key=lambda x: (x.get("type") != "precise", -x.get("ratio", 0.0)))
    return matches


# ─── proactive offer ─────────────────────────────────────────────────────

def _load_offer_state() -> dict:
    if not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_offer_state(state: dict) -> None:
    try:
        _atomic_write_json(_STATE_FILE, state)
    except Exception as e:
        print(f"  [pattern_learning] offer-state save failed: {e}")


def _compose_offer_line(p: dict) -> str:
    action = p.get("action", "")
    offer  = p.get("offer") or _offer_for(action)
    arg    = (p.get("common_arg") or "").strip()
    # Specialise: if we know the usual play_music argument (e.g. "michael
    # jackson"), tailor the offer line to it — that's the spec's headline
    # example ("Shall I queue your usual Michael Jackson mix, sir?").
    if action in ("play_music", "youtube_play", "spotify", "apple_music") and arg:
        return f"Shall I queue your usual {_titlecase(arg)} mix, sir? You typically start it about now."
    return offer


def _titlecase(s: str) -> str:
    small = {"and", "or", "of", "the", "in", "on", "for", "to", "a", "an"}
    parts = s.split()
    out = []
    for i, w in enumerate(parts):
        if i > 0 and w in small:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def maybe_pattern_offer_v2(bypass_throttle: bool = False) -> str:
    """Return ONE JARVIS-style proactive offer if a prediction matches the
    current moment AND we haven't surfaced this same prediction key today.
    Returns '' when nothing matches.

    Reads from the aggregator snapshot (data/usage_patterns_aggregated.json).
    Self-throttles via data/usage_patterns_offer_state.json — at-most-once
    per day per prediction key, with 90-day pruning of stale state entries.
    """
    snapshot = _load_aggregated()
    if not snapshot:
        return ""
    matches = predictions_for_now(snapshot)
    if not matches:
        return ""

    today = time.strftime("%Y-%m-%d", time.localtime())
    state = _load_offer_state() if not bypass_throttle else {}

    for p in matches:
        key = p.get("key", "")
        if not key:
            continue
        if not bypass_throttle and state.get(key) == today:
            continue
        line = _compose_offer_line(p)
        if not line:
            continue
        if not bypass_throttle:
            state[key] = today
            _prune_state(state)
            _save_offer_state(state)
        return line
    return ""


def _prune_state(state: dict) -> None:
    try:
        cutoff = time.strftime(
            "%Y-%m-%d",
            time.localtime(time.time() - 90 * 86400),
        )
        for k in list(state.keys()):
            v = state.get(k)
            if isinstance(v, str) and v < cutoff:
                del state[k]
    except Exception:
        pass


# ─── weekly digest (habit clusters by day-of-week × hour-window) ─────────

_DOW_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday")


def _monday_of(ts: float) -> str:
    """ISO YYYY-MM-DD of the Monday on or before `ts`'s local-clock date.
    Used to label cached weekly_summaries rows."""
    lt = time.localtime(ts)
    today = _dt.date(lt.tm_year, lt.tm_mon, lt.tm_mday)
    monday = today - _dt.timedelta(days=today.weekday())
    return monday.isoformat()


def _format_hour_band(start_h: int) -> str:
    """'8 PM' for hour 20, '8–10 PM' for a 2-hour window starting at 20."""
    end_h = (start_h + WEEKLY_HOUR_WINDOW) % 24
    def _fmt(h: int) -> str:
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12} {suffix}"
    return f"{_fmt(start_h)}–{_fmt(end_h)}"


def _cluster_label(dow: int, hour_start: int, action: str,
                   weeks_seen: int, weeks_total: int,
                   common_arg: str = "") -> str:
    """Human-readable summary line ('Friday 8–10 PM: Netflix 4/4 weeks').
    Mirrors the spec's example phrasing."""
    day = _DOW_NAMES[dow % 7]
    verb = _verb_for(action)
    tail = ""
    if common_arg:
        tail = f" ({_titlecase(common_arg)})"
    return (f"{day} {_format_hour_band(hour_start)}: "
            f"{verb}{tail} {weeks_seen}/{weeks_total} weeks")


def _cluster_offer(dow: int, hour_start: int, action: str,
                   common_arg: str = "") -> str:
    """Anticipatory spoken offer for the cluster. Tailored for media actions
    so we can reproduce the spec's 'Friday Netflix' headline. Falls back to
    the generic action verb otherwise."""
    day = _DOW_NAMES[dow % 7]
    band = _format_hour_band(hour_start).lower()
    if action == "netflix":
        return (f"Sir, it's {day} — and around {band} you usually queue "
                f"Netflix. Shall I bring it up?")
    if action in ("play_music", "resume_music", "apple_music", "spotify"):
        if common_arg:
            return (f"Sir, your usual {_titlecase(common_arg)} listening "
                    f"window is upon us. Shall I queue it?")
        return ("Sir, this is normally when you settle into music. "
                "Shall I queue your usual playlist?")
    if action == "check_teams":
        return (f"Sir, you typically check Teams around {band} on {day}s — "
                f"shall I take a look?")
    if action == "morning_briefing":
        return f"Sir, your usual {day} morning briefing — shall I deliver it?"
    if action == "evening_briefing":
        return f"Sir, your usual {day} evening briefing — shall I deliver it?"
    verb = _verb_for(action)
    return (f"Sir, it's {day} around {band} — you usually {verb}. "
            "Shall I proceed?")


def compute_weekly_digest(now: float | None = None) -> dict:
    """Cluster the last WEEKLY_LOOKBACK_DAYS of events by (dow, hour-window,
    action) and return the top WEEKLY_MAX_CARDS clusters with confidence
    scores. Persists the result to the weekly_summaries table keyed by this
    week's Monday so the consumer skill can read a stable snapshot.

    Returns a dict with shape:
      {
        "week_start":   "YYYY-MM-DD",
        "computed_at":  <epoch>,
        "lookback_days": int,
        "clusters":     [ {dow, hour_start, action, weeks_seen,
                           weeks_total, occurrences, common_arg,
                           label, offer, key, confidence}, … ]
      }
    """
    now = float(now if now is not None else time.time())
    week_start = _monday_of(now)
    cutoff_ts  = now - WEEKLY_LOOKBACK_DAYS * 86400.0

    digest = {
        "week_start":    week_start,
        "computed_at":   now,
        "lookback_days": WEEKLY_LOOKBACK_DAYS,
        "clusters":      [],
    }

    conn = _connect_db()
    rows: list[tuple] = []
    if conn is not None:
        try:
            with _db_lock:
                rows = conn.execute(
                    "SELECT ts, dow, hour, action, arg FROM events "
                    "WHERE ts >= ?",
                    (cutoff_ts,),
                ).fetchall()
        except Exception as e:
            print(f"  [pattern_learning] weekly digest SQL failed: {e}")
            rows = []
        finally:
            try: conn.close()
            except Exception: pass  # pragma: no cover - defensive close in weekly-digest SQL finally

    # SQL was empty or unavailable — fall back to the JSONL so first-boot
    # systems with no SQLite mirror yet still produce a digest.
    if not rows:
        for e in _load_events():
            ts  = e.get("ts")
            dow = e.get("wd")
            hr  = e.get("hour")
            act = e.get("action")
            arg = e.get("arg") or ""
            if not (isinstance(ts, (int, float)) and isinstance(dow, int)
                    and isinstance(hr, int) and isinstance(act, str)):
                continue
            if ts < cutoff_ts:
                continue
            rows.append((float(ts), dow, hr, act, arg))

    if not rows:
        _save_weekly_digest(digest)
        return digest

    # Bucket events: key = (dow, hour_start_aligned, action)
    # hour_start_aligned snaps the event's hour to the bottom of its
    # WEEKLY_HOUR_WINDOW band (so 20:45 → band 20-22; 21:10 → band 20-22).
    buckets: dict[tuple[int, int, str], dict] = {}
    weeks_in_lookback: set[str] = set()
    for ts, dow, hr, action, arg in rows:
        try:
            week_label = _monday_of(float(ts))
        except Exception:
            continue
        weeks_in_lookback.add(week_label)
        band = (int(hr) // WEEKLY_HOUR_WINDOW) * WEEKLY_HOUR_WINDOW
        key = (int(dow), band, str(action))
        slot = buckets.setdefault(key, {
            "occurrences": 0,
            "weeks":       set(),
            "args":        Counter(),
        })
        slot["occurrences"] += 1
        slot["weeks"].add(week_label)
        a = (arg or "").strip().lower()
        if a:
            slot["args"][a] += 1

    weeks_total = max(1, len(weeks_in_lookback))

    clusters: list[dict] = []
    for (dow, hour_start, action), data in buckets.items():
        occ = data["occurrences"]
        weeks_seen = len(data["weeks"])
        if occ < WEEKLY_MIN_OCCURRENCES:
            continue
        if weeks_seen < WEEKLY_MIN_WEEKS:
            continue
        common_arg = data["args"].most_common(1)[0][0] if data["args"] else ""
        # Confidence: fraction of distinct weeks this slot fired in. A habit
        # that happens 3/3 weeks ranks above one that happens 3/4.
        confidence = weeks_seen / weeks_total
        clusters.append({
            "key":         f"weekly|{dow}|{hour_start}|{action}",
            "dow":         dow,
            "dow_name":    _DOW_NAMES[dow % 7],
            "hour_start":  hour_start,
            "hour_end":    (hour_start + WEEKLY_HOUR_WINDOW) % 24,
            "action":      action,
            "common_arg":  common_arg,
            "occurrences": occ,
            "weeks_seen":  weeks_seen,
            "weeks_total": weeks_total,
            "confidence":  round(confidence, 3),
            "label":       _cluster_label(dow, hour_start, action,
                                          weeks_seen, weeks_total, common_arg),
            "offer":       _cluster_offer(dow, hour_start, action, common_arg),
        })

    clusters.sort(key=lambda c: (c["confidence"], c["occurrences"]), reverse=True)
    digest["clusters"] = clusters[:WEEKLY_MAX_CARDS]
    _save_weekly_digest(digest)
    return digest


def _save_weekly_digest(digest: dict) -> None:
    conn = _connect_db()
    if conn is None:
        return
    try:
        with _db_lock:
            conn.execute(
                "INSERT OR REPLACE INTO weekly_summaries(week_start, computed_at, cluster_data) "
                "VALUES (?, ?, ?)",
                (digest["week_start"], digest["computed_at"],
                 json.dumps(digest.get("clusters", []), ensure_ascii=False)),
            )
    except Exception as e:
        print(f"  [pattern_learning] weekly digest save failed: {e}")
    finally:
        try: conn.close()
        except Exception: pass  # pragma: no cover - defensive close in weekly-digest save finally


def load_latest_weekly_digest() -> dict:
    """Return the most recently computed weekly digest, or {} if none cached.
    Public API consumed by skills/weekly_digest_briefing.py."""
    conn = _connect_db()
    if conn is None:
        return {}
    try:
        with _db_lock:
            row = conn.execute(
                "SELECT week_start, computed_at, cluster_data "
                "FROM weekly_summaries "
                "ORDER BY computed_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return {}
        try:
            clusters = json.loads(row[2] or "[]")
            if not isinstance(clusters, list):
                clusters = []
        except Exception:
            clusters = []
        return {
            "week_start":  row[0],
            "computed_at": float(row[1] or 0.0),
            "clusters":    clusters,
        }
    except Exception as e:
        print(f"  [pattern_learning] weekly digest load failed: {e}")
        return {}
    finally:
        try: conn.close()
        except Exception: pass  # pragma: no cover - defensive close in weekly-digest load finally


# ─── background scheduler ────────────────────────────────────────────────

def _aggregator_loop() -> None:
    time.sleep(INITIAL_DELAY_SECONDS)
    # Bootstrap: if there's no cached snapshot, or it's stale, run aggregation now.
    try:
        agg = _load_aggregated()
        last = float(agg.get("generated_at", 0.0) or 0.0)
        if not last or (time.time() - last) > MAX_AGGREGATE_AGE_SECS:
            aggregate()
    except Exception as e:
        print(f"  [pattern_learning] initial aggregate failed: {e}")

    # Bootstrap weekly digest as well so the consumer skill has something to
    # render on first boot. _connect_db() handles schema init lazily.
    try:
        latest = load_latest_weekly_digest()
        age = (time.time() - float(latest.get("computed_at", 0.0) or 0.0)) \
            if latest else float("inf")
        if not latest or age > WEEKLY_AGE_MAX_SECS:
            compute_weekly_digest()
    except Exception as e:
        print(f"  [pattern_learning] initial weekly digest failed: {e}")

    while True:
        try:
            time.sleep(LOOP_INTERVAL_SECONDS)
            lt = time.localtime()
            agg = _load_aggregated()
            last = float(agg.get("generated_at", 0.0) or 0.0)
            age  = time.time() - last if last else float("inf")
            # Nightly slot at NIGHTLY_HOUR ± LOOP_INTERVAL — and refuse to
            # rerun if we did so within the past 20 hours.
            if lt.tm_hour == NIGHTLY_HOUR and age > 20 * 3600:
                print("  [pattern_learning] nightly aggregation kicking in")
                aggregate()
            # Weekly digest slot: Monday morning at WEEKLY_DIGEST_HOUR, with a
            # 6-day minimum gap so we never run twice in the same week (cheap
            # guard against DST or local-time rollovers).
            if (lt.tm_wday == WEEKLY_DIGEST_DOW
                    and lt.tm_hour == WEEKLY_DIGEST_HOUR):
                latest = load_latest_weekly_digest()
                last_w = float(latest.get("computed_at", 0.0) or 0.0) if latest else 0.0
                if (time.time() - last_w) > 6 * 86400:
                    print("  [pattern_learning] weekly digest computation kicking in")
                    compute_weekly_digest()
        except Exception as e:
            print(f"  [pattern_learning] scheduler error: {e}")


# ─── actions registered with JARVIS ──────────────────────────────────────

def _act_pattern_predictions(_: str = "") -> str:
    snapshot = _load_aggregated()
    if not snapshot:
        return "No predictions yet, sir — the aggregator hasn't built a snapshot."
    n_events = snapshot.get("events", 0)
    span = snapshot.get("days_span", 0.0)
    broad   = snapshot.get("broad",   []) or []
    precise = snapshot.get("precise", []) or []
    if not broad and not precise:
        return (f"No strong patterns yet, sir — {n_events} events across "
                f"{span:.1f} days. Need at least {MIN_DAYS_OBSERVED} days "
                f"of data.")
    # Render top 5 across both lists, precise first.
    chosen = (precise + broad)[:5]
    body = "; ".join(p["label"] for p in chosen if p.get("label"))
    return f"From {n_events} events over {span:.1f} days, sir: {body}."


def _act_pattern_offer_now(_: str = "") -> str:
    line = maybe_pattern_offer_v2(bypass_throttle=True)
    return line or "No prediction matches the current moment, sir."


def _act_pattern_aggregate(_: str = "") -> str:
    snapshot = aggregate()
    n_broad   = len(snapshot.get("broad",   []) or [])
    n_precise = len(snapshot.get("precise", []) or [])
    return (f"Aggregation complete, sir — {snapshot.get('events', 0)} events, "
            f"{n_broad} broad pattern{'s' if n_broad != 1 else ''}, "
            f"{n_precise} precise pattern{'s' if n_precise != 1 else ''}.")


def _act_weekly_digest(_: str = "") -> str:
    """Recompute the weekly digest right now and read back the top clusters
    as a single JARVIS-style line. Manual trigger for the user; the
    background scheduler also fires this on Mondays at WEEKLY_DIGEST_HOUR."""
    digest = compute_weekly_digest()
    clusters = digest.get("clusters") or []
    if not clusters:
        return ("No weekly habits have stabilised yet, sir — need a couple "
                "of weeks of consistent data before patterns emerge.")
    top = clusters[:3]
    body = "; ".join(c["label"] for c in top if c.get("label"))
    return f"Weekly digest, sir — {body}."


def _act_pattern_stats(_: str = "") -> str:
    snapshot = _load_aggregated()
    n_events = snapshot.get("events", 0) if snapshot else 0
    span     = snapshot.get("days_span", 0.0) if snapshot else 0.0
    last     = float(snapshot.get("generated_at", 0.0) or 0.0) if snapshot else 0.0
    if last:
        age_min = int((time.time() - last) // 60)
        age_str = (f"{age_min} minute{'s' if age_min != 1 else ''} ago"
                   if age_min < 120
                   else f"{age_min // 60} hour{'s' if age_min // 60 != 1 else ''} ago")
    else:
        age_str = "never"
    n_broad   = len(snapshot.get("broad",   []) or []) if snapshot else 0
    n_precise = len(snapshot.get("precise", []) or []) if snapshot else 0
    state     = _load_offer_state()
    today     = time.strftime("%Y-%m-%d", time.localtime())
    offered   = sum(1 for v in state.values() if v == today)
    return (f"Pattern learning, sir: {n_events} events across {span:.1f} days; "
            f"{n_broad} broad, {n_precise} precise; last aggregation {age_str}; "
            f"{offered} offer{'s' if offered != 1 else ''} surfaced today.")


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["pattern_predictions"] = _act_pattern_predictions
    actions["pattern_offer_now"]   = _act_pattern_offer_now
    actions["pattern_aggregate"]   = _act_pattern_aggregate
    actions["pattern_stats"]       = _act_pattern_stats
    actions["weekly_digest"]       = _act_weekly_digest

    _ensure_data_dir()
    # One-shot: replay JSONL into SQLite on the first boot after this upgrade
    # so the weekly digest has historical data instead of waiting a week for
    # SQLite to fill organically. No-op once events table is populated.
    try:
        _backfill_sqlite_from_jsonl_if_empty()
    except Exception as e:
        print(f"  [pattern_learning] backfill skipped: {e}")
    # Guard against duplicate loops on skill reload (load_skills re-execs the
    # module → fresh globals, so only an OS-thread name check survives).
    if any(th.name == "pattern-aggregator" and th.is_alive()
           for th in threading.enumerate()):
        print("  [pattern_learning] aggregator already running — skipping "
              "duplicate (skill reload)")
        return
    t = threading.Thread(target=_aggregator_loop, daemon=True,
                         name="pattern-aggregator")
    t.start()
    print(f"  [pattern_learning] event log: {os.path.relpath(_LOG_FILE, _HERE)}; "
          f"sqlite: {os.path.relpath(_DB_FILE, _HERE)}; "
          f"nightly @ {NIGHTLY_HOUR:02d}:00 local; "
          f"weekly @ {_DOW_NAMES[WEEKLY_DIGEST_DOW]} "
          f"{WEEKLY_DIGEST_HOUR:02d}:00 local")


# ─── offline smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover - manual offline smoke test entry point
    # Generate synthetic events covering a 21-day window so we can exercise
    # both broad-window and precise-clock detection without touching real data.
    print("Running offline smoke test…")
    fake: list[dict] = []
    base = time.time() - 21 * 86400
    for day in range(21):
        day_ts = base + day * 86400
        lt = time.localtime(day_ts)
        wd = lt.tm_wday
        date = time.strftime("%Y-%m-%d", lt)
        dow  = time.strftime("%A", lt)
        # play_music between 9-11am on weekdays (we'll vary it Mon-Fri)
        if wd < 5 and day % 5 != 4:    # 4/5 weekdays = 80%
            h = 9 + (day % 2)
            fake.append({"ts": day_ts, "iso": "", "date": date, "dow": dow,
                         "wd": wd, "hour": h, "min": 17,
                         "action": "play_music", "arg": "michael jackson"})
        # check_teams at 09:15 ± 5 min daily
        fake.append({"ts": day_ts, "iso": "", "date": date, "dow": dow,
                     "wd": wd, "hour": 9, "min": 13 + (day % 5),
                     "action": "check_teams", "arg": ""})
    # Write to a temp jsonl, point _LOG_FILE at it, aggregate, restore.
    tmpfd, tmppath = tempfile.mkstemp(suffix=".jsonl")
    os.close(tmpfd)
    with open(tmppath, "w", encoding="utf-8") as f:
        for e in fake:
            f.write(json.dumps(e) + "\n")
    orig_log = _LOG_FILE
    orig_agg = _AGG_FILE
    orig_state = _STATE_FILE
    globals()["_LOG_FILE"]   = tmppath
    globals()["_AGG_FILE"]   = tmppath + ".agg.json"
    globals()["_STATE_FILE"] = tmppath + ".state.json"
    try:
        snap = aggregate()
        print("Broad predictions:")
        for p in snap["broad"]:
            print("  ", p["label"], f"(ratio {p['ratio']}, occ {p['occurrences']})")
        print("Precise predictions:")
        for p in snap["precise"]:
            print("  ", p["label"], f"(ratio {p['ratio']}, sd {p['stddev_min']})")
        print("Offer (bypass throttle):", repr(maybe_pattern_offer_v2(bypass_throttle=True)))
    finally:
        for p in (tmppath, tmppath + ".agg.json", tmppath + ".state.json"):
            try: os.unlink(p)
            except Exception: pass
        globals()["_LOG_FILE"]   = orig_log
        globals()["_AGG_FILE"]   = orig_agg
        globals()["_STATE_FILE"] = orig_state
