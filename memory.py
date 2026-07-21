"""Command-level pattern memory for JARVIS.

Logs every voice command with timestamp + day-of-week, then derives
observed habits (e.g. 'user plays Michael Jackson 73% of Friday evenings')
that can be surfaced as proactive in-character offers.

This is complementary to bobert_companion.save_session_pattern, which
records session-level aggregates. Here we record per-command granularity
so we can mine specific targets (artist names, app names, search terms)
that get repeated at the same time of week.

Storage:
  memory/voice_commands.jsonl     — one JSON object per line
  memory/pattern_offers_state.json — once-per-day-per-pattern throttle
  memory/session_summaries.json   — queryable index of LLM session summaries

Public API:
  record_voice_command(text)              — log accepted user utterance
  forget_voice_commands_since(cutoff_ts)  — purge logged commands in a window
  get_patterns()                          — list of detected habit dicts
  maybe_pattern_offer()                   — JARVIS-style offer string
  record_session_summary(summary, ...)    — append one session-level summary
  get_session_summaries(query, ...)       — sessions matching a time phrase
  parse_time_reference(text)              — phrase → (start, end[, hours])
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict

_HERE          = os.path.dirname(os.path.abspath(__file__))

# Blue/green: a staging JARVIS sets JARVIS_STAGING=1 in its env so this
# module redirects all its persistent state into data_staging/memory/
# instead of the live memory/ directory. Prevents test traffic from
# corrupting the real pattern bank during a smoke-test cycle.
if os.environ.get("JARVIS_STAGING", "").strip() == "1":
    _MEM_DIR       = os.path.join(_HERE, "data_staging", "memory")
    _BOBERT_MEMORY = os.path.join(_HERE, "data_staging", "bobert_memory.json")
else:
    _MEM_DIR       = os.path.join(_HERE, "memory")
    _BOBERT_MEMORY = os.path.join(_HERE, "bobert_memory.json")

_LOG_FILE      = os.path.join(_MEM_DIR, "voice_commands.jsonl")
_OFFER_STATE   = os.path.join(_MEM_DIR, "pattern_offers_state.json")
_SESSION_FILE  = os.path.join(_MEM_DIR, "session_summaries.json")

MAX_LOG_ENTRIES         = 5000   # cap file size; keep most-recent N
PATTERN_MIN_OCCURRENCES = 3      # min repetitions before we call it a pattern
PATTERN_MIN_RATIO       = 0.50   # target must dominate its (dow,bucket) slot
PATTERN_USUAL_RATIO     = 0.75   # ratio at which an offer collapses to "the usual"
APP_PATTERN_MIN_RATIO   = 0.40   # apps repeat across many command types — lower bar
MIN_DAYS_OF_HISTORY     = 14     # don't surface anticipations until 14 days logged
DEBOUNCE_SECONDS        = 2.0    # ignore duplicate text within this window

_log_lock                  = threading.Lock()
_last_command: dict        = {"text": "", "ts": 0.0}
_writes_since_rotate: list = [0]


def _localtime(ts: float | None = None) -> time.struct_time:
    return time.localtime(ts if ts is not None else time.time())


def _bucket_hour(hour: int) -> str:
    if 5 <= hour < 12:  return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 22: return "evening"
    return "night"


def _bucket_label(bucket: str) -> str:
    return bucket  # already user-readable


def _ensure_dir() -> None:
    try:
        os.makedirs(_MEM_DIR, exist_ok=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

# Browser titles look like "Page — Google Chrome" / "GitHub - Mozilla Firefox";
# the part after the last " - " / " — " is usually the app name. Anything in
# this set is treated as the canonical app label so titles collapse together
# across documents.
_APP_TITLE_TAILS = {
    "google chrome":          "Chrome",
    "chrome":                 "Chrome",
    "mozilla firefox":        "Firefox",
    "firefox":                "Firefox",
    "microsoft edge":         "Edge",
    "edge":                   "Edge",
    "visual studio code":     "VS Code",
    "vs code":                "VS Code",
    "code":                   "VS Code",
    "microsoft teams":        "Teams",
    "teams":                  "Teams",
    "discord":                "Discord",
    "slack":                  "Slack",
    "spotify":                "Spotify",
    "apple music":            "Apple Music",
    "itunes":                 "iTunes",
    "youtube":                "YouTube",
    "youtube music":          "YouTube Music",
    "bambu studio":           "Bambu Studio",
    "orcaslicer":             "OrcaSlicer",
    "prusaslicer":            "PrusaSlicer",
    "autodesk fusion 360":    "Fusion 360",
    "fusion 360":             "Fusion 360",
    "fusion":                 "Fusion 360",
    "blender":                "Blender",
    "powershell":             "PowerShell",
    "windows powershell":     "PowerShell",
    "windows terminal":       "Windows Terminal",
    "notepad":                "Notepad",
    "outlook":                "Outlook",
    "microsoft outlook":      "Outlook",
    "explorer":               "Explorer",
    "file explorer":          "Explorer",
}


def _get_active_app() -> str:
    """Best-effort canonical name of the currently-focused window's owning
    app. Used to enrich voice-command logs so we can detect 'user typically
    has Teams focused around this time'. Returns '' on any failure."""
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return ""
    try:
        w = gw.getActiveWindow()
        if w is None:
            return ""
        title = (getattr(w, "title", "") or "").strip()
        if not title:
            return ""
        # Try the most common "Doc - App" / "Doc — App" suffix shape first.
        for sep in (" — ", " - ", " – "):
            if sep in title:
                tail = title.rsplit(sep, 1)[-1].strip().lower()
                if tail in _APP_TITLE_TAILS:
                    return _APP_TITLE_TAILS[tail]
        # No recognised suffix: maybe the whole title IS the app name.
        whole = title.lower()
        if whole in _APP_TITLE_TAILS:
            return _APP_TITLE_TAILS[whole]
        # Walk the recognised list and accept a substring match so e.g.
        # "Inbox — you — Outlook" still resolves to Outlook.
        for key, label in _APP_TITLE_TAILS.items():
            if key in whole:
                return label
        # Fallback: truncate the raw title — useful as a weak signal but
        # won't aggregate well across sessions.
        return title[:60]
    except Exception:
        return ""


def record_voice_command(text: str, active_app: str | None = None) -> None:
    """Append one entry to memory/voice_commands.jsonl. Cheap; safe to call
    on every accepted user utterance. Skips blanks, bare wake-words, and
    rapid duplicates within DEBOUNCE_SECONDS.

    `active_app` defaults to the currently-focused window's app name so the
    log can support 'you typically check Teams around now' anticipations.
    Callers may pass an explicit value (or '') to override or disable."""
    if not text:
        return
    cleaned = text.strip()
    if not cleaned:
        return

    lowered = cleaned.lower().strip(" ,.!?")
    if lowered in {"jarvis", "hey jarvis", "ok jarvis", "okay jarvis",
                   "yes", "no", "stop", "cancel"}:
        return

    now = time.time()
    with _log_lock:
        if cleaned == _last_command["text"] and (now - _last_command["ts"]) < DEBOUNCE_SECONDS:
            return
        _last_command["text"] = cleaned
        _last_command["ts"]   = now

    if active_app is None:
        active_app = _get_active_app()

    lt = _localtime(now)
    entry = {
        "ts":   now,
        "iso":  time.strftime("%Y-%m-%dT%H:%M:%S", lt),
        "dow":  time.strftime("%A", lt),
        "hour": lt.tm_hour,
        "text": cleaned[:300],
        "app":  (active_app or "")[:80],
    }
    _ensure_dir()
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [pattern_memory] log write failed: {e}")
        return

    _writes_since_rotate[0] += 1
    if _writes_since_rotate[0] >= 100:
        _writes_since_rotate[0] = 0
        _maybe_rotate()


def forget_voice_commands_since(cutoff_ts: float) -> int:
    """Drop voice-command log entries recorded at or after ``cutoff_ts``
    (epoch seconds). voice_commands.jsonl holds the user's verbatim last-hour
    speech, so a 'forget the last hour' that skipped it left the exact
    material the owner asked to erase (2026-07-21 audit — same gap as the
    LTM episode log, one store over).

    Atomic rewrite via the same tmp + os.replace pattern as _maybe_rotate,
    under _log_lock. Entries without a parseable ts are KEPT (legacy
    convention: old entries survive a time-window forget). Exceptions
    propagate so the caller can disclose a failed purge. Returns the number
    of entries dropped."""
    with _log_lock:
        if not os.path.exists(_LOG_FILE):
            return 0
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        keep = []
        dropped = 0
        for line in lines:
            ts = None
            try:
                ts = float(json.loads(line).get("ts"))
            except Exception:
                ts = None           # unparseable / ts-less → treated as old
            if ts is not None and ts >= cutoff_ts:
                dropped += 1
                continue
            keep.append(line)
        if dropped:
            tmp = _LOG_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(keep)
            os.replace(tmp, _LOG_FILE)
        return dropped


def _maybe_rotate() -> None:
    # Hold _log_lock across the read + rewrite so this rotation is atomic w.r.t.
    # anything else that takes _log_lock: two rotations can't race, and the
    # debounce bookkeeping in record_voice_command can't interleave with the
    # os.replace swap. The caller has already released _log_lock before calling
    # this, so re-acquiring here cannot deadlock.
    # NOTE: the JSONL append in record_voice_command is intentionally outside
    # _log_lock (kept that way to avoid holding the lock across file I/O on the
    # hot path), so this does not fully close the append-vs-rotate window; it
    # does serialise the rotate itself, which was previously unprotected.
    try:
        with _log_lock:
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


def _load_entries() -> list[dict]:
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


# ─────────────────────────────────────────────────────────────────────────────
#  Target extraction — turn a free-text utterance into a comparable
#  (category, target) pair so repetitions across days actually collapse.
# ─────────────────────────────────────────────────────────────────────────────

_PLAY_RE   = re.compile(
    r"\b(?:play|put on|queue up|queue|start playing|start)\s+(.+?)"
    r"(?:\s+(?:on|in|please|now|next|after|by|from|for me)\b|[?!.]|$)",
    re.I,
)
_OPEN_RE   = re.compile(
    r"\b(?:open|launch|start|fire up|bring up|pull up)\s+(.+?)"
    r"(?:\s+(?:on|in|please|now|for me)\b|[?!.]|$)",
    re.I,
)
_SEARCH_RE = re.compile(
    r"\b(?:search(?:\s+for)?|look up|google|find|search the web for)\s+(.+?)"
    r"(?:\s+(?:on|in|please|now)\b|[?!.]|$)",
    re.I,
)
_SCREEN_RE = re.compile(
    r"\b(?:see (?:my |the )?screen|"
    r"what(?:'s| is) on (?:my |the )?screen|"
    r"screen(?:shot)?|see my display|check (?:my |the )?screen)\b",
    re.I,
)
_TIMER_RE  = re.compile(
    r"\b(?:set (?:a |the )?timer|timer for|remind me (?:in|to))\b",
    re.I,
)

_JUNK_TARGETS = {
    "music", "something", "anything", "a song", "the song",
    "the music", "some music", "a playlist", "a movie", "a show",
    "it", "that", "this", "now", "please", "sir", "for me",
    "app", "the app", "the file", "a file", "the page",
}


def _normalize_target(raw: str) -> str:
    t = raw.strip().lower().rstrip(".,!?")
    t = re.sub(r"\s+", " ", t)
    return t


def _extract_target(text: str) -> tuple[str, str] | None:
    """Return (category, target) or None.

    Categories:
      'play'    — play X (music/track/album/artist)
      'open'    — open/launch X (app, site)
      'search'  — search/google X
      'screen'  — generic screen-vision request (target='')
      'timer'   — generic timer/reminder (target='')
    """
    s = text.strip()

    m = _PLAY_RE.search(s)
    if m:
        target = _normalize_target(m.group(1))
        if target and target not in _JUNK_TARGETS and len(target) <= 80:
            return ("play", target)

    m = _OPEN_RE.search(s)
    if m:
        target = _normalize_target(m.group(1))
        if target and target not in _JUNK_TARGETS and len(target) <= 80:
            return ("open", target)

    m = _SEARCH_RE.search(s)
    if m:
        target = _normalize_target(m.group(1))
        if target and target not in _JUNK_TARGETS and len(target) <= 80:
            return ("search", target)

    if _SCREEN_RE.search(s):
        return ("screen", "")
    if _TIMER_RE.search(s):
        return ("timer", "")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Pattern derivation
# ─────────────────────────────────────────────────────────────────────────────

def _history_span_days(entries: list[dict]) -> float:
    """How many days separate the oldest and newest entries (0 if not enough
    data). Used by the 14-day anticipation gate so JARVIS doesn't make
    confident pattern offers based on three days of usage."""
    timestamps = [e.get("ts") for e in entries
                  if isinstance(e.get("ts"), (int, float))]
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)) / 86400.0


def get_patterns(min_days: float | None = None) -> list[dict]:
    """Return list of detected habits, sorted strongest-first.

    Each entry:
      {
        "category": "play"|"open"|"search"|"screen"|"timer"|"app_usage",
        "target":   "michael jackson",   # for app_usage this is the app name
        "dow":      "Friday",
        "bucket":   "evening",
        "hour":     19,                  # avg of matching events
        "ratio":    0.73,                # fraction of slot events
        "count":    11,                  # times target seen in slot
        "total":    15,                  # total recognised events in slot
        "key":      "play|michael jackson|Friday|evening",
        "summary":  "user plays michael jackson 73% of Friday evenings"
      }

    Pass `min_days=0` to bypass the 14-day history gate (used by tests and
    by callers that just want raw detection regardless of confidence)."""
    entries = _load_entries()
    if len(entries) < PATTERN_MIN_OCCURRENCES:
        return []

    if min_days is None:
        min_days = MIN_DAYS_OF_HISTORY
    if min_days > 0 and _history_span_days(entries) < min_days:
        return []

    slot_events: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    slot_hours:  dict[tuple[str, str], list[int]]             = defaultdict(list)
    # App-focused-at-command-time tally — separate from `slot_events` because
    # the same app frequently appears across many command categories, and we
    # want to surface that ambient pattern ('you typically check Teams around
    # now') even when no specific verb dominates.
    slot_apps:   dict[tuple[str, str], list[str]]             = defaultdict(list)

    for e in entries:
        text = e.get("text", "")
        dow  = e.get("dow",  "")
        hour = e.get("hour", -1)
        if not dow or not isinstance(hour, int) or hour < 0 or hour > 23:
            continue
        bucket = _bucket_hour(hour)
        extracted = _extract_target(text)
        if extracted is not None:
            slot_events[(dow, bucket)].append(extracted)
            slot_hours[(dow, bucket)].append(hour)
        app = (e.get("app") or "").strip()
        if app:
            slot_apps[(dow, bucket)].append(app)
            # App usage also counts toward the slot's hour profile so the
            # representative hour stays accurate when only ambient signals fire.
            slot_hours[(dow, bucket)].append(hour)

    patterns: list[dict] = []
    for (dow, bucket), events in slot_events.items():
        total = len(events)
        if total < PATTERN_MIN_OCCURRENCES:
            continue
        counter = Counter(events)
        for (cat, target), count in counter.most_common():
            if count < PATTERN_MIN_OCCURRENCES:
                break
            ratio = count / total
            if ratio < PATTERN_MIN_RATIO:
                continue
            hours = slot_hours[(dow, bucket)]
            repr_hour = int(round(sum(hours) / len(hours))) if hours else 0
            if target:
                summary = (f"user {cat}s {target} {int(round(ratio * 100))}% "
                           f"of {dow} {_bucket_label(bucket)}s")
            else:
                summary = (f"user typically uses {cat} during {dow} "
                           f"{_bucket_label(bucket)}s "
                           f"({count}/{total} times)")
            patterns.append({
                "category": cat,
                "target":   target,
                "dow":      dow,
                "bucket":   bucket,
                "hour":     repr_hour,
                "ratio":    round(ratio, 3),
                "count":    count,
                "total":    total,
                "key":      f"{cat}|{target}|{dow}|{bucket}",
                "summary":  summary,
            })

    # App-usage pattern type — 'you typically check Teams around now'.
    for (dow, bucket), apps in slot_apps.items():
        total_apps = len(apps)
        if total_apps < PATTERN_MIN_OCCURRENCES:
            continue
        for app_name, count in Counter(apps).most_common():
            if count < PATTERN_MIN_OCCURRENCES:
                break
            ratio = count / total_apps
            if ratio < APP_PATTERN_MIN_RATIO:
                continue
            hours = slot_hours[(dow, bucket)]
            repr_hour = int(round(sum(hours) / len(hours))) if hours else 0
            summary = (f"user typically has {app_name} focused "
                       f"{int(round(ratio * 100))}% of "
                       f"{dow} {_bucket_label(bucket)}s")
            patterns.append({
                "category": "app_usage",
                "target":   app_name,
                "dow":      dow,
                "bucket":   bucket,
                "hour":     repr_hour,
                "ratio":    round(ratio, 3),
                "count":    count,
                "total":    total_apps,
                "key":      f"app_usage|{app_name}|{dow}|{bucket}",
                "summary":  summary,
            })

    patterns.sort(key=lambda p: (p["ratio"], p["count"]), reverse=True)
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
#  Proactive offer — surface a pattern at most once per day per pattern key
# ─────────────────────────────────────────────────────────────────────────────

def _load_offer_state() -> dict:
    if not os.path.exists(_OFFER_STATE):
        return {}
    try:
        with open(_OFFER_STATE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_offer_state(state: dict) -> None:
    _ensure_dir()
    try:
        tmp = _OFFER_STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _OFFER_STATE)
    except Exception:
        pass


def _title(target: str) -> str:
    # Title-case but leave short connector words lowercase for readability
    small = {"and", "or", "of", "the", "in", "on", "for", "to", "a", "an"}
    parts = target.split()
    out = []
    for i, w in enumerate(parts):
        if i > 0 and w in small:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _format_hour_12h(hour: int) -> str:
    """24-hour int → 'H PM' / 'H AM' for spoken phrasing ('It's Friday 9 PM')."""
    try:
        h = int(hour)
    except Exception:
        return ""
    if not (0 <= h <= 23):
        return ""
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12} {suffix}"


def _phrase_for(pattern: dict) -> str:
    cat    = pattern["category"]
    target = pattern["target"]
    dow    = pattern["dow"]
    bucket = pattern["bucket"]
    label  = _bucket_label(bucket)
    hour   = pattern.get("hour", -1)
    ratio  = pattern.get("ratio", 0.0)

    # When we have a representative hour, prefer the more specific
    # "It's Friday 9 PM, sir" phrasing over the bucket-only fallback.
    hour_str = _format_hour_12h(hour) if isinstance(hour, int) else ""
    when = f"{dow} {hour_str}" if hour_str else f"{dow} {label}"

    # Very high confidence collapses to the spec's "shall I queue the usual?"
    # — only kicks in for play patterns, since "the usual" implies media.
    usual = ratio >= PATTERN_USUAL_RATIO

    if cat == "play" and target:
        if usual:
            return f"It's {when}, sir — shall I queue the usual?"
        return (f"It's {when}, sir — shall I queue the "
                f"{_title(target)} essentials?")
    if cat == "open" and target:
        return (f"It's {when}, sir — shall I open {_title(target)} "
                f"for you?")
    if cat == "search" and target:
        return (f"It's {when}, sir — would you like me to pull up "
                f"{_title(target)}?")
    if cat == "screen":
        return (f"It's {when}, sir — you usually ask me to check "
                f"the screen around now. Shall I take a look?")
    if cat == "timer":
        return (f"It's {when}, sir — you usually set a timer around "
                f"now. Shall I start one?")
    if cat == "app_usage" and target:
        # 'You typically check Teams around now — anything I should peek at?'
        return (f"You typically check {target} around now, sir — "
                f"anything you'd like me to peek at?")
    return ""


def maybe_pattern_offer(min_days: float | None = None) -> str:
    """Return a JARVIS-style proactive offer if a strong pattern matches the
    current time-of-week AND we haven't surfaced it yet today. Otherwise ''.

    Safe to call repeatedly: the persistent state file (pattern_offers_state.json)
    guarantees at-most-once-per-day per pattern key.

    The 14-day history gate inside `get_patterns()` ensures JARVIS doesn't
    confidently anticipate behaviour after only a few days of data. Pass
    `min_days=0` to override (useful for diagnostics / manual triggers).

    Consults the optional pattern_learning skill FIRST — it operates on
    action-level events (data/usage_patterns.jsonl) with a separate aggregator
    that supports broad-window and precise-clock predictions. If the skill
    isn't loaded, or has nothing matching, we fall back to the voice-utterance
    patterns mined here."""
    import sys as _sys
    _pl = _sys.modules.get("skill_pattern_learning")
    if _pl is not None:
        try:
            v2 = _pl.maybe_pattern_offer_v2()
            if v2:
                return v2
        except Exception as _e:
            print(f"  [pattern_memory] v2 offer failed: {_e}")

    patterns = get_patterns(min_days=min_days)
    if not patterns:
        return ""

    now = _localtime()
    cur_dow    = time.strftime("%A", now)
    cur_bucket = _bucket_hour(now.tm_hour)

    matching = [p for p in patterns
                if p["dow"] == cur_dow and p["bucket"] == cur_bucket]
    if not matching:
        return ""

    state = _load_offer_state()
    today = time.strftime("%Y-%m-%d", now)

    for p in matching:
        if state.get(p["key"]) == today:
            continue
        phrase = _phrase_for(p)
        if not phrase:
            continue
        state[p["key"]] = today
        # Prune state entries older than 90 days so the file doesn't grow
        # without bound.
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
        _save_offer_state(state)
        return phrase
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Session summaries — queryable index for `session_memory_recall`
#
#  Each entry mirrors the LLM-generated one-sentence summary that
#  bobert_companion.save_session_to_memory writes into bobert_memory.json,
#  plus enough timestamp metadata for natural-language time filtering
#  ('yesterday', 'last night', 'this morning', 'monday').
# ─────────────────────────────────────────────────────────────────────────────

MAX_SESSION_ENTRIES = 500   # cap file size; keep most recent N

_session_lock = threading.Lock()


def _load_sessions_file() -> list[dict]:
    if not os.path.exists(_SESSION_FILE):
        return []
    try:
        with open(_SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_sessions_file(entries: list[dict]) -> None:
    _ensure_dir()
    try:
        tmp = _SESSION_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries[-MAX_SESSION_ENTRIES:], f, indent=2)
        os.replace(tmp, _SESSION_FILE)
    except Exception as e:
        print(f"  [pattern_memory] session save failed: {e}")


def _load_legacy_sessions() -> list[dict]:
    """Pre-existing session summaries live in bobert_memory.json["sessions"]
    with shape {date, location, summary}. Bring them in (read-only) as a
    fallback so a fresh session_summaries.json isn't empty on day one."""
    if not os.path.exists(_BOBERT_MEMORY):
        return []
    try:
        with open(_BOBERT_MEMORY, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for s in data.get("sessions", []):
            if not isinstance(s, dict):
                continue
            date = s.get("date", "")
            summary = s.get("summary", "").strip()
            if not date or not summary:
                continue
            # Best-effort day-of-week from the date
            try:
                dow = _dt.date.fromisoformat(date).strftime("%A")
            except Exception:
                dow = ""
            out.append({
                "date":         date,
                "day":          dow,
                "hour_started": -1,
                "hour_ended":   -1,
                "location":     s.get("location", ""),
                "summary":      summary,
                "source":       "bobert_memory",
            })
        return out
    except Exception:
        return []


def record_session_summary(summary: str,
                           start_ts: float | None = None,
                           end_ts:   float | None = None,
                           location: str = "") -> None:
    """Append (or update) one session-level summary entry. Safe to call from a
    shutdown hook OR a periodic checkpoint — atomic write, no LLM calls.

    Idempotent per session: entries are keyed by iso_start. If an entry with
    the same iso_start already exists, it is REPLACED rather than duplicated.
    This is what makes the periodic in-session checkpoint safe — calling this
    every few minutes during one session updates the single entry instead of
    appending a near-duplicate each time. The checkpoint exists because the old
    shutdown-only writer almost never ran: JARVIS is usually force-killed,
    crashes, or is killed by the upgrade pipeline, all of which bypass the
    clean-exit hook (root cause of the 3-day-stale empty session_summaries.json
    found in the 2026-05-30 audit)."""
    if not summary or not summary.strip():
        return
    end_ts   = end_ts or time.time()
    start_ts = start_ts or end_ts
    start_lt = _localtime(start_ts)
    end_lt   = _localtime(end_ts)
    iso_start = time.strftime("%Y-%m-%dT%H:%M:%S", start_lt)
    entry = {
        "ts":           end_ts,
        "iso_start":    iso_start,
        "iso_end":      time.strftime("%Y-%m-%dT%H:%M:%S", end_lt),
        "date":         time.strftime("%Y-%m-%d", start_lt),
        "day":          time.strftime("%A",       start_lt),
        "hour_started": start_lt.tm_hour,
        "hour_ended":   end_lt.tm_hour,
        "location":     location,
        "summary":      summary.strip()[:1000],
    }
    with _session_lock:
        entries = _load_sessions_file()
        # Replace any existing entry for THIS session (same iso_start) so
        # repeated checkpoints update one row instead of stacking duplicates.
        entries = [e for e in entries
                   if e.get("iso_start") != iso_start]
        entries.append(entry)
        _save_sessions_file(entries)


# ── Natural-language time reference parser ──────────────────────────────────

_DAYS = ("monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday")


def _today() -> _dt.date:
    return _dt.date.today()


def parse_time_reference(text: str
                         ) -> tuple[_dt.date, _dt.date, tuple[int, int] | None]:
    """Parse a phrase like 'yesterday' / 'last night' / 'monday' into a
    (start_date, end_date_inclusive, hour_range_or_None) triple.

    Falls back to (today - 7 days, today, None) so an unconstrained query
    like 'what did we do this week' still returns recent sessions."""
    t = (text or "").lower()
    today = _today()

    # "X days ago" — Whisper transcribes spoken numerals, so accept either
    # digits or the spelled-out one-through-twelve range.
    _NUM_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "a couple of": 2, "a couple": 2, "a few": 3,
    }
    m = re.search(r"\b(\d{1,3})\s+days?\s+ago\b", t)
    if m:
        n = int(m.group(1))
        d = today - _dt.timedelta(days=n)
        return (d, d, None)
    m = re.search(r"\b(" + "|".join(re.escape(k) for k in _NUM_WORDS) + r")\s+days?\s+ago\b", t)
    if m:
        n = _NUM_WORDS[m.group(1)]
        d = today - _dt.timedelta(days=n)
        return (d, d, None)

    if re.search(r"\btoday\b|\bthis session\b", t):
        # "this morning" / "this afternoon" / "this evening" narrows the hour
        if "morning" in t:
            return (today, today, (5, 11))
        if "afternoon" in t:
            return (today, today, (12, 16))
        if "evening" in t or "tonight" in t:
            return (today, today, (17, 23))
        return (today, today, None)

    if "yesterday" in t:
        d = today - _dt.timedelta(days=1)
        if "morning" in t:
            return (d, d, (5, 11))
        if "afternoon" in t:
            return (d, d, (12, 16))
        if "evening" in t or "night" in t:
            return (d, d, (17, 23))
        return (d, d, None)

    if "last night" in t:
        # Pragmatic reading: "last night" usually means yesterday evening.
        # Pre-dawn sessions of today are better captured by "this morning"
        # or "earlier", which we handle separately above/below.
        d_prev = today - _dt.timedelta(days=1)
        return (d_prev, d_prev, (17, 23))

    if "this morning" in t:
        return (today, today, (5, 11))
    if "this afternoon" in t:
        return (today, today, (12, 16))
    if "this evening" in t or "tonight" in t:
        return (today, today, (17, 23))

    if "last week" in t:
        # ISO week starts Monday — pull the previous 7-day Mon..Sun
        wd = today.weekday()                   # 0=Mon, 6=Sun
        this_mon = today - _dt.timedelta(days=wd)
        last_mon = this_mon - _dt.timedelta(days=7)
        last_sun = this_mon - _dt.timedelta(days=1)
        return (last_mon, last_sun, None)

    if "this week" in t:
        wd = today.weekday()
        this_mon = today - _dt.timedelta(days=wd)
        return (this_mon, today, None)

    # Named weekday → most recent occurrence (in the past 7 days)
    for i, name in enumerate(_DAYS):
        if re.search(rf"\b{name}\b", t):
            target_wd = i
            delta = (today.weekday() - target_wd) % 7
            if delta == 0:
                # ambiguous — could mean today or the prior week
                # default to today since "this monday" + today=monday means today
                if "last " + name in t:
                    delta = 7
            d = today - _dt.timedelta(days=delta)
            # let "monday evening" etc. still narrow hours
            if "morning" in t:
                return (d, d, (5, 11))
            if "afternoon" in t:
                return (d, d, (12, 16))
            if "evening" in t or "night" in t:
                return (d, d, (17, 23))
            return (d, d, None)

    if re.search(r"\b(?:the other day|recently|lately|earlier)\b", t):
        return (today - _dt.timedelta(days=7), today, None)

    # Default fallback: last week's worth of sessions
    return (today - _dt.timedelta(days=7), today, None)


def _entry_matches_window(entry: dict,
                          start: _dt.date,
                          end:   _dt.date,
                          hours: tuple[int, int] | None) -> bool:
    date_str = entry.get("date", "")
    try:
        d = _dt.date.fromisoformat(date_str)
    except Exception:
        return False
    if d < start or d > end:
        return False
    if hours is not None:
        h = entry.get("hour_started", -1)
        if not isinstance(h, int) or h < 0:
            # No hour info on legacy entries — only allow them through when
            # the window is broad (not an hour-specific question).
            return False
        lo, hi = hours
        if lo <= hi:
            if not (lo <= h <= hi):
                return False
        else:
            # crossover (e.g. 22..5)
            if not (h >= lo or h <= hi):
                return False
    return True


def get_session_summaries(query: str = "",
                          limit: int = 10,
                          include_legacy: bool = True) -> list[dict]:
    """Return session summaries matching `query`'s time reference.

    Entries are sorted newest-first and truncated to `limit`. If the
    queryable index is empty (or shy a few entries), legacy session
    summaries from bobert_memory.json["sessions"] are pulled in so the
    feature works on the very first call after install."""
    with _session_lock:
        entries = list(_load_sessions_file())
    if include_legacy:
        existing = {(e.get("date"), e.get("summary")) for e in entries}
        for legacy in _load_legacy_sessions():
            key = (legacy.get("date"), legacy.get("summary"))
            if key not in existing:
                entries.append(legacy)
                existing.add(key)

    start, end, hours = parse_time_reference(query)
    matched = [e for e in entries if _entry_matches_window(e, start, end, hours)]

    # Newest first — use iso_end if available, else date as a fallback
    def _key(e: dict) -> str:
        return e.get("iso_end") or e.get("iso_start") or e.get("date") or ""
    matched.sort(key=_key, reverse=True)
    return matched[:limit]


def describe_window(query: str) -> str:
    """Render the resolved time window in JARVIS-readable English (used by
    the recall action when no sessions matched, so we can be specific:
    'I have no recollection from yesterday evening, sir.')."""
    start, end, hours = parse_time_reference(query)
    t = (query or "").lower()
    if "last night" in t:           return "from last night"
    if "yesterday" in t:
        if "morning" in t:          return "from yesterday morning"
        if "afternoon" in t:        return "from yesterday afternoon"
        if "evening" in t:          return "from yesterday evening"
        return "from yesterday"
    if "this morning" in t:         return "from this morning"
    if "this afternoon" in t:       return "from this afternoon"
    if "this evening" in t or "tonight" in t: return "from this evening"
    if "today" in t:                return "from today"
    if "last week" in t:            return "from last week"
    if "this week" in t:            return "from this week"
    for name in _DAYS:
        if re.search(rf"\b{name}\b", t):
            return f"from {name.capitalize()}"
    if start == end:
        return f"from {start.isoformat()}"
    return f"between {start.isoformat()} and {end.isoformat()}"


# ─────────────────────────────────────────────────────────────────────────────
#  Offline smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "play michael jackson",
        "play michael jackson please",
        "play michael jackson on apple music",
        "set a timer for ten minutes",
        "what's on my screen",
        "open chrome",
        "google python tutorials",
        "search for the best espresso machine",
        "JARVIS",
        "yes",
    ]
    for s in samples:
        print(f"{s!r:55} → {_extract_target(s)}")
    print()
    print(f"Active app (live): {_get_active_app()!r}")
    print(f"History span (days): {_history_span_days(_load_entries()):.1f} "
          f"(gate at {MIN_DAYS_OF_HISTORY})")
    print()
    print("Patterns (no history gate):",
          json.dumps(get_patterns(min_days=0), indent=2))
    print("Patterns (14-day gate):",
          json.dumps(get_patterns(), indent=2))
    print("Offer (no gate):", repr(maybe_pattern_offer(min_days=0)))
    print("Offer (gated):  ", repr(maybe_pattern_offer()))

    print()
    print("--- session_memory_recall time-reference parser ---")
    for q in [
        "what did we do yesterday",
        "remind me what I was working on last night",
        "what happened this morning",
        "what did we work on monday",
        "anything from last week",
        "three days ago",
        "what was I doing",
    ]:
        s, e, h = parse_time_reference(q)
        print(f"  {q!r:55} → {s}..{e} hours={h}")
    print()
    print("Sessions (default window):")
    for s in get_session_summaries(""):
        print(f"  {s.get('date')} ({s.get('day','?')}) — {s.get('summary','')[:80]}")
