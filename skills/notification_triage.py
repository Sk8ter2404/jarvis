"""
Notification triage center.

Hooks the Windows ``UserNotificationListener`` API (``winsdk`` /
``winrt-runtime``) to capture every toast notification the system pops —
Teams DMs, Outlook arrivals, Slack mentions, calendar reminders, Steam
friend chatter, Github / build / CI alerts, all of it — and routes each
one through a tunable rules engine. Each rule fires one of four actions:

  read_aloud   — proactive_announce() the toast verbatim (or via a
                 lightly cleaned-up "<app> notification: <text>" form)
  log          — capture in data/notifications/<date>.jsonl, no audio
  drop         — silent, not even logged (used for marketing / steam
                 friend online / generic OS upgrade nags)
  classify     — defer to the Haiku triage classifier (the fallthrough
                 default when no explicit rule matches)

Rules live in ``notification_rules.json`` at the project root (created on
first boot from DEFAULT_RULES if missing). Each rule is a dict with:
  - id          short slug, must be unique
  - match       {app_pattern, title_pattern, body_pattern}   regex, all opt
  - action      one of read_aloud | log | drop | classify
  - priority    optional integer; higher wins when multiple rules match
  - reason      free-text describing why the rule exists (read by the
                voice action list_notification_rules)

The Haiku classifier is invoked only when no rule matches AND
``ENABLE_LLM_CLASSIFIER`` is True AND ``ANTHROPIC_API_KEY`` is set. It
returns a single token (``urgent`` / ``fyi`` / ``newsletter`` / ``spam``)
which is mapped to read_aloud / log / log / drop respectively. A
positive classification gets cached for SNOOZE_SECONDS so a Slack
channel that just pinged 4 times in a row doesn't burn 4 API calls.

When ``dnd_focus_mode.is_focus_mode_active()`` returns True we
SUPPRESS read_aloud — the toast still lands in the log but JARVIS keeps
quiet. Critical rules (priority >= 100) bypass focus mode so "Sir, the
build failed" still cuts through.

Actions registered (voice triggers via the dispatcher):
  triage_status                 — summary of subsystem health, rule
                                  count, recent activity, focus state.
  list_notification_rules       — speak the active rules.
  add_notification_rule <json>  — append a rule. JSON arg shape:
                                  {"id":"slack_general","match":{...},
                                   "action":"drop","reason":"too noisy"}
  remove_notification_rule <id> — delete by id.
  recent_notifications_summary [N]
                                — read back the last N captured toasts
                                  (default 5, max 20). Alias:
                                  list_recent_notifications.
  pause_notification_triage     — stop the listener until resumed.
  resume_notification_triage    — restart polling.

Module-level helper for other skills:
  recent_notifications(n=10) -> list[dict]   for HUDs / status_panel.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import traceback

# Project-root onto sys.path so `core.atomic_io` resolves whether this
# module is loaded as `skills.notification_triage` or run standalone.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

try:
    from core.atomic_io import _atomic_write_json
except Exception:  # pragma: no cover — core may be mid-import at boot
    import tempfile

    def _atomic_write_json(path, data, *, indent=2):
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

_log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 4.0          # how often we sweep UserNotificationListener
INITIAL_DELAY_SECONDS = 20.0         # let the rest of JARVIS settle before listening
ASYNC_OP_TIMEOUT_SECONDS = 5.0       # ceiling for the manual IAsyncOperation poll — a
                                     # wedged WinRT op must not spin the daemon forever
SNOOZE_SECONDS        = 300          # don't re-speak / re-classify identical toast within 5 min
MAX_NOTIFICATION_LOG  = 250          # recent_notifications ring buffer size in memory
MAX_SPOKEN_BODY_CHARS = 220          # truncate long bodies before TTS
ENABLE_LLM_CLASSIFIER = True         # set False to skip Haiku and just `log` unmatched
LLM_MODEL             = "claude-haiku-4-5"
LLM_TIMEOUT_SECONDS   = 6.0
HIGH_PRIORITY_FLOOR   = 100          # rules with priority >= this bypass focus mode

_RULES_FILE     = os.path.join(_PROJECT_DIR, "notification_rules.json")
_DATA_DIR       = os.path.join(_PROJECT_DIR, "data", "notifications")

DEFAULT_RULES: list[dict] = [
    # ── HIGH-priority — always read aloud, even in focus mode ──────────
    {
        "id": "build_or_ci_failure",
        "match": {
            "title_pattern": r"(?i)\b(build|deploy|pipeline|ci|workflow|job)\b.*\b(fail|error|broken|red)\b"
                             r"|\b(fail|error)\b.*\b(build|deploy|pipeline|ci|workflow|job)\b",
        },
        "action": "read_aloud",
        "priority": 120,
        "reason": "Build / CI failures cut through focus mode.",
    },
    {
        "id": "calendar_reminder",
        "match": {
            "app_pattern": r"(?i)(outlook|teams|calendar)",
            "title_pattern": r"(?i)\b(reminder|upcoming|starting|in \d+ min)\b",
        },
        "action": "read_aloud",
        "priority": 110,
        "reason": "Meeting reminders are time-sensitive.",
    },
    {
        "id": "bambu_print_event",
        "match": {
            "app_pattern": r"(?i)(bambu|bambustudio)",
        },
        "action": "read_aloud",
        "priority": 105,
        "reason": "Print finished / failed events should always interrupt.",
    },
    # ── Normal priority — read aloud, but quiet during focus mode ──────
    {
        "id": "teams_dm",
        "match": {
            "app_pattern": r"(?i)teams",
            "title_pattern": r"(?i)^(?!.*\bchannel\b).+",   # exclude channel pings
        },
        "action": "read_aloud",
        "priority": 50,
        "reason": "Direct messages on Teams.",
    },
    {
        "id": "outlook_new_mail",
        "match": {
            "app_pattern": r"(?i)outlook",
        },
        "action": "classify",
        "priority": 40,
        "reason": "Hand new mail to Haiku for urgent/fyi/newsletter triage.",
    },
    # ── Silent log ─────────────────────────────────────────────────────
    {
        "id": "slack_channel",
        "match": {
            "app_pattern": r"(?i)slack",
            "title_pattern": r"(?i)#",
        },
        "action": "log",
        "priority": 30,
        "reason": "Slack channel pings: logged but not spoken.",
    },
    # ── Drop entirely (silent) ─────────────────────────────────────────
    {
        "id": "steam_friend_online",
        "match": {
            "app_pattern": r"(?i)steam",
            "title_pattern": r"(?i)(now playing|online|just logged|achievement)",
        },
        "action": "drop",
        "priority": 20,
        "reason": "Steam friend online / achievement spam.",
    },
    {
        "id": "marketing",
        "match": {
            "body_pattern": r"(?i)\b(unsubscribe|sale|promo|% off|deal of the day|"
                             r"limited time|sign up now|free trial|coupon)\b",
        },
        "action": "drop",
        "priority": 15,
        "reason": "Marketing toasts.",
    },
    {
        "id": "windows_upgrade_nag",
        "match": {
            "app_pattern": r"(?i)(windows\s*update|microsoft store|edge|onedrive)",
            "title_pattern": r"(?i)(restart|update|upgrade|sign in|try .*free|recommend)",
        },
        "action": "drop",
        "priority": 10,
        "reason": "Generic OS / Edge / Store nags.",
    },
]

LLM_VERDICTS_TO_ACTION = {
    "urgent":     "read_aloud",
    "fyi":        "log",
    "newsletter": "log",
    "spam":       "drop",
}

# ─── State ───────────────────────────────────────────────────────────────
_state_lock           = threading.RLock()
_rules: list[dict]    = []
_seen_ids: set[int]   = set()                       # WinRT notification ids we've processed
_recent: list[dict]   = []                          # in-memory ring buffer
_snooze: dict[str, float] = {}                      # dedupe key → last-fire ts

# task-75: persist _seen_ids and _snooze across bounces so a fresh JARVIS
# doesn't replay every Windows notification that arrived in the last few
# minutes (the WinRT listener catches the unread backlog on connect).
import os as _os, json as _json, tempfile as _tempfile, time as _time
_PROJECT_DIR_NT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DEDUPE_FILE    = _os.path.join(_PROJECT_DIR_NT, "data", "notification_dedup.json")
_DEDUPE_TTL_SEC = 6 * 3600   # 6h — long enough to survive a bounce, short
                             #     enough to recover if a stuck id is poisoned


def _load_dedupe_state() -> None:
    """At import: hydrate _seen_ids + _snooze from disk if recent enough."""
    if not _os.path.exists(_DEDUPE_FILE):
        return
    try:
        with open(_DEDUPE_FILE, "r", encoding="utf-8") as f:
            d = _json.load(f)
        if not isinstance(d, dict):
            return
        cutoff = _time.time() - _DEDUPE_TTL_SEC
        if d.get("ts", 0) < cutoff:
            return
        sids = d.get("seen_ids") or []
        with _state_lock:
            for s in sids:
                try: _seen_ids.add(int(s))
                except Exception: pass
            snz = d.get("snooze") or {}
            if isinstance(snz, dict):
                for k, v in snz.items():
                    try:
                        if float(v) >= cutoff:
                            _snooze[str(k)] = float(v)
                    except Exception: pass
    except Exception:
        pass


def _save_dedupe_state() -> None:
    """Atomic write of dedupe state. Best-effort — a failed write just
    means the next bounce might re-announce up to a few minutes of
    notifications, which is much better than the current 'every bounce'."""
    try:
        d = _os.path.dirname(_DEDUPE_FILE)
        _os.makedirs(d, exist_ok=True)
        with _state_lock:
            payload = {
                "ts":       _time.time(),
                "seen_ids": list(_seen_ids)[-2000:],   # cap at 2000 ids
                "snooze":   dict(_snooze),
            }
        fd, tmp = _tempfile.mkstemp(dir=d, prefix=".notif_", suffix=".tmp")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            _os.replace(tmp, _DEDUPE_FILE)
        except Exception:
            try: _os.remove(tmp)
            except Exception: pass
    except Exception:
        pass


_load_dedupe_state()
_last_dedupe_save = [0.0]


# ─── Content-based dedupe (survives WinRT-ID regeneration) ───────────────
#
# WinRT reissues notification ids after a watchdog stall/restart, so the
# id-based set above misses backlog repeats. A second cache keyed on
# (sender, content_hash, app) catches "a DM alerting on every session
# restart" — same body text, same sender, same app, suppressed for 4h.
_CONTENT_DEDUPE_FILE    = _os.path.join(_PROJECT_DIR_NT, "data",
                                        "notification_content_dedup.json")
_CONTENT_DEDUPE_TTL_SEC = 4 * 3600   # 4h

_content_dedupe: dict[str, float] = {}    # content_key → last-seen ts


def _content_dedupe_key(app: str, title: str, body: str) -> str:
    """Stable key for content-based dedup. Sender = first word of title
    (Teams/Slack DM titles lead with sender). Hash covers title+body so
    same-sender, different-message cases (calendar reminders, repeated
    digests) don't collide."""
    title_norm = " ".join((title or "").split())
    body_norm  = " ".join((body or "").split())
    sender = title_norm.split(maxsplit=1)[0].lower() if title_norm else ""
    content_hash = hashlib.md5(
        f"{title_norm.lower()}\n{body_norm.lower()}".encode("utf-8")
    ).hexdigest()
    app_norm = (app or "").strip().lower()
    return f"{app_norm}|{sender}|{content_hash}"


def _prune_content_dedupe(now: float | None = None) -> None:
    if now is None:
        now = _time.time()
    cutoff = now - _CONTENT_DEDUPE_TTL_SEC
    with _state_lock:
        stale = [k for k, ts in _content_dedupe.items() if ts < cutoff]
        for k in stale:
            _content_dedupe.pop(k, None)


def _load_content_dedupe() -> None:
    """Hydrate the content cache from disk. Prunes stale entries on load
    so a long-idle restart doesn't suppress fresh notifications."""
    if not _os.path.exists(_CONTENT_DEDUPE_FILE):
        return
    try:
        with open(_CONTENT_DEDUPE_FILE, "r", encoding="utf-8") as f:
            d = _json.load(f)
        if not isinstance(d, dict):
            return
        entries = d.get("entries")
        if not isinstance(entries, dict):
            return
        cutoff = _time.time() - _CONTENT_DEDUPE_TTL_SEC
        with _state_lock:
            for k, v in entries.items():
                try:
                    ts = float(v)
                except Exception:
                    continue
                if ts >= cutoff:
                    _content_dedupe[str(k)] = ts
    except Exception:
        pass


def _save_content_dedupe() -> None:
    """Atomic write of the content dedupe cache. Best-effort — a failed
    write costs at most one re-announce of the same toast after the next
    restart."""
    try:
        d = _os.path.dirname(_CONTENT_DEDUPE_FILE)
        _os.makedirs(d, exist_ok=True)
        _prune_content_dedupe()
        with _state_lock:
            payload = {
                "ts":      _time.time(),
                "entries": dict(_content_dedupe),
            }
        fd, tmp = _tempfile.mkstemp(dir=d, prefix=".notif_content_",
                                    suffix=".tmp")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            _os.replace(tmp, _CONTENT_DEDUPE_FILE)
        except Exception:
            try: _os.remove(tmp)
            except Exception: pass
    except Exception:
        pass


_load_content_dedupe()


# ─── Timestamp-bucket dedupe (catches rapid Teams/Windows retries) ───────
#
# Teams and the Windows toast subsystem occasionally resend the same
# notification multiple times within seconds (network blip, focus-mode
# toggle, app or watchdog restart). The 4-hour _content_dedupe above
# handles the long-horizon "a DM after every session restart" case
# but uses app-name normalization that misses Teams' inconsistent display
# names ("Microsoft Teams" vs "Teams classic" vs "Teams (work or school)").
# This tighter cache keys on (sender, body_snippet, timestamp_bucket) so
# a rapid retry burst collapses to a single announcement without
# depending on app identity.
_NOTIF_TS_DEDUPE_FILE     = _os.path.join(_PROJECT_DIR_NT, "data",
                                          "notification_timestamp_dedup.json")
_NOTIF_TS_DEDUPE_TTL_SEC  = 10 * 60   # 10 minutes
_NOTIF_TS_BUCKET_SECONDS  = 60        # 1-minute bucket — catches rapid retries
_NOTIF_TS_BODY_SNIPPET_CH = 120       # first 120 chars of normalized body

_notification_timestamp_dedup: dict[str, float] = {}    # key → first-seen ts


def _timestamp_bucket(ts: float) -> int:
    """Convert a Unix timestamp into a coarse bucket index. Same-bucket
    notifications fall into the same dedup key; cross-bucket ones are
    treated as fresh announcements (the existing 4-hour content cache
    still suppresses sender-level cross-window repeats)."""
    return int(ts // _NOTIF_TS_BUCKET_SECONDS)


def _body_snippet(body: str, title: str = "") -> str:
    """Normalize the first ~120 chars of the body for stable hashing.
    Falls back to the title when the body is empty so empty-body toasts
    from different senders don't collide on the empty-string hash."""
    text = (body or "").strip()
    if not text:
        text = (title or "").strip()
    norm = " ".join(text.split()).lower()
    return norm[:_NOTIF_TS_BODY_SNIPPET_CH]


def _notification_timestamp_dedupe_key(title: str, body: str,
                                       bucket: int) -> str:
    """Stable dedup key. Sender is the first word of the title (Teams /
    Outlook / Slack DM titles lead with the sender name). Snippet is
    hashed so the key length stays bounded and the on-disk cache is
    cheap to serialize."""
    title_norm = " ".join((title or "").split())
    sender = title_norm.split(maxsplit=1)[0].lower() if title_norm else ""
    snippet = _body_snippet(body, title)
    snippet_hash = hashlib.md5(snippet.encode("utf-8")).hexdigest()
    return f"{sender}|{snippet_hash}|{bucket}"


def _prune_notification_timestamp_dedup(now: float | None = None) -> None:
    if now is None:
        now = _time.time()
    cutoff = now - _NOTIF_TS_DEDUPE_TTL_SEC
    with _state_lock:
        stale = [k for k, ts in _notification_timestamp_dedup.items()
                 if ts < cutoff]
        for k in stale:
            _notification_timestamp_dedup.pop(k, None)


def _load_notification_timestamp_dedup() -> None:
    """Hydrate the timestamp-bucket cache from disk. Prunes stale entries
    on load so a long-idle restart doesn't suppress fresh notifications."""
    if not _os.path.exists(_NOTIF_TS_DEDUPE_FILE):
        return
    try:
        with open(_NOTIF_TS_DEDUPE_FILE, "r", encoding="utf-8") as f:
            d = _json.load(f)
        if not isinstance(d, dict):
            return
        entries = d.get("entries")
        if not isinstance(entries, dict):
            return
        cutoff = _time.time() - _NOTIF_TS_DEDUPE_TTL_SEC
        with _state_lock:
            for k, v in entries.items():
                try:
                    ts = float(v)
                except Exception:
                    continue
                if ts >= cutoff:
                    _notification_timestamp_dedup[str(k)] = ts
    except Exception:
        pass


def _save_notification_timestamp_dedup() -> None:
    """Atomic write of the timestamp-bucket dedup cache. Best-effort — a
    failed write costs at most one re-announce of the same toast after
    the next restart."""
    try:
        d = _os.path.dirname(_NOTIF_TS_DEDUPE_FILE)
        _os.makedirs(d, exist_ok=True)
        _prune_notification_timestamp_dedup()
        with _state_lock:
            payload = {
                "ts":      _time.time(),
                "entries": dict(_notification_timestamp_dedup),
            }
        fd, tmp = _tempfile.mkstemp(dir=d, prefix=".notif_ts_",
                                    suffix=".tmp")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            _os.replace(tmp, _NOTIF_TS_DEDUPE_FILE)
        except Exception:
            try: _os.remove(tmp)
            except Exception: pass
    except Exception:
        pass


_load_notification_timestamp_dedup()


# ─── Announce-specific dedupe (SHA-1, 10-min TTL) ────────────────────────
#
# Even with the WinRT-id set, the content cache, and the timestamp-bucket
# cache, JARVIS was still re-announcing the same Teams/email ping every
# poll cycle. The earlier caches all suppress at the listener-entry layer
# and depend on either app-name normalization, sender heuristics, or
# bucket alignment — any one of them can miss, and the read_aloud branch
# below trusts that everything upstream already deduped.
#
# This cache sits directly in front of _proactive_announce(): a SHA-1
# hash of (source, sender, normalized_body), 10-minute TTL, persisted
# atomically. If the same announcement fires twice within the window
# we skip the speak call regardless of what the upstream caches did.
_ANNOUNCE_DEDUPE_FILE    = _os.path.join(_PROJECT_DIR_NT, "data",
                                         "notification_announce_dedup.json")
_ANNOUNCE_DEDUPE_TTL_SEC = 10 * 60   # 10 minutes

_announce_dedup: dict[str, float] = {}    # sha1 hex → last-announce ts


def _announce_dedup_key(source: str, sender: str, body: str) -> str:
    """SHA-1 hash of normalized (source, sender, body). SHA-1 (not MD5)
    so we don't collide with the MD5-based _content_dedupe / _notification_
    timestamp_dedup keyspaces during troubleshooting."""
    source_norm = " ".join((source or "").split()).lower()
    sender_norm = " ".join((sender or "").split()).lower()
    body_norm   = " ".join((body or "").split()).lower()
    payload = f"{source_norm}\x1f{sender_norm}\x1f{body_norm}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _prune_announce_dedup(now: float | None = None) -> None:
    if now is None:
        now = _time.time()
    cutoff = now - _ANNOUNCE_DEDUPE_TTL_SEC
    with _state_lock:
        stale = [k for k, ts in _announce_dedup.items() if ts < cutoff]
        for k in stale:
            _announce_dedup.pop(k, None)


def _load_announce_dedup() -> None:
    """Hydrate the announce-dedup cache from disk. Prunes stale entries on
    load so a long-idle restart doesn't suppress fresh announcements."""
    if not _os.path.exists(_ANNOUNCE_DEDUPE_FILE):
        return
    try:
        with open(_ANNOUNCE_DEDUPE_FILE, "r", encoding="utf-8") as f:
            d = _json.load(f)
        if not isinstance(d, dict):
            return
        entries = d.get("entries")
        if not isinstance(entries, dict):
            return
        cutoff = _time.time() - _ANNOUNCE_DEDUPE_TTL_SEC
        with _state_lock:
            for k, v in entries.items():
                try:
                    ts = float(v)
                except Exception:
                    continue
                if ts >= cutoff:
                    _announce_dedup[str(k)] = ts
    except Exception:
        pass


def _save_announce_dedup() -> None:
    """Atomic write of the announce-dedup cache. Best-effort — a failed
    write costs at most one re-announce of the same toast after the next
    restart."""
    try:
        d = _os.path.dirname(_ANNOUNCE_DEDUPE_FILE)
        _os.makedirs(d, exist_ok=True)
        _prune_announce_dedup()
        with _state_lock:
            payload = {
                "ts":      _time.time(),
                "entries": dict(_announce_dedup),
            }
        fd, tmp = _tempfile.mkstemp(dir=d, prefix=".notif_announce_",
                                    suffix=".tmp")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            _os.replace(tmp, _ANNOUNCE_DEDUPE_FILE)
        except Exception:
            try: _os.remove(tmp)
            except Exception: pass
    except Exception:
        pass


_load_announce_dedup()


# ─── Announce-specific SHA-256 dedupe (window-scoped, 30-min TTL) ────────
#
# Logs showed JARVIS still re-announcing the same Teams/Windows toast on
# every poll cycle in cases where the SHA-1 announce_dedup above missed —
# typically when the upstream caches mutated their normalization (focus
# mode toggling app names, Teams reissuing ids) so the SHA-1 key drifted
# while the toast was effectively the same.
#
# This cache hashes (sender, body, window_id) with SHA-256 — a different
# keyspace and hash algorithm from every upstream cache, so a collision
# in one cannot poison the other. window_id is the WinRT notification id
# (the closest analog the UserNotificationListener API exposes; the
# planner flagged this as a clarification point and id is the only
# stable per-window identifier available). 30-min TTL is intentionally
# longer than the SHA-1 cache's 10 min: this is the last line of defense
# specifically against the "every poll cycle" bug.
_ANNOUNCE_SHA256_DEDUPE_FILE    = _os.path.join(_PROJECT_DIR_NT, "data",
                                                "notification_announce_sha256_dedup.json")
_ANNOUNCE_SHA256_DEDUPE_TTL_SEC = 30 * 60   # 30 minutes

_announce_sha256_dedup: dict[str, float] = {}    # sha256 hex → last-announce ts


def _announce_sha256_dedupe_key(sender: str, body: str,
                                window_id: object) -> str:
    """SHA-256 of (sender, body, window_id). Distinct from the SHA-1
    announce_dedup and the MD5 content/timestamp caches so collisions
    are independent. window_id may be int, str, or 0/None — coerced to
    str for stable hashing."""
    sender_norm = " ".join((sender or "").split()).lower()
    body_norm   = " ".join((body or "").split()).lower()
    win_norm    = str(window_id if window_id is not None else "")
    payload = f"{sender_norm}\x1f{body_norm}\x1f{win_norm}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prune_announce_sha256_dedup(now: float | None = None) -> None:
    if now is None:
        now = _time.time()
    cutoff = now - _ANNOUNCE_SHA256_DEDUPE_TTL_SEC
    with _state_lock:
        stale = [k for k, ts in _announce_sha256_dedup.items() if ts < cutoff]
        for k in stale:
            _announce_sha256_dedup.pop(k, None)


def _load_announce_sha256_dedup() -> None:
    """Hydrate the SHA-256 announce-dedup cache from disk. Prunes stale
    entries on load so a long-idle restart doesn't suppress fresh
    announcements."""
    if not _os.path.exists(_ANNOUNCE_SHA256_DEDUPE_FILE):
        return
    try:
        with open(_ANNOUNCE_SHA256_DEDUPE_FILE, "r", encoding="utf-8") as f:
            d = _json.load(f)
        if not isinstance(d, dict):
            return
        entries = d.get("entries")
        if not isinstance(entries, dict):
            return
        cutoff = _time.time() - _ANNOUNCE_SHA256_DEDUPE_TTL_SEC
        with _state_lock:
            for k, v in entries.items():
                try:
                    ts = float(v)
                except Exception:
                    continue
                if ts >= cutoff:
                    _announce_sha256_dedup[str(k)] = ts
    except Exception:
        pass


def _save_announce_sha256_dedup() -> None:
    """Atomic write of the SHA-256 announce-dedup cache. Best-effort — a
    failed write costs at most one re-announce of the same toast after
    the next restart."""
    try:
        d = _os.path.dirname(_ANNOUNCE_SHA256_DEDUPE_FILE)
        _os.makedirs(d, exist_ok=True)
        _prune_announce_sha256_dedup()
        with _state_lock:
            payload = {
                "ts":      _time.time(),
                "entries": dict(_announce_sha256_dedup),
            }
        fd, tmp = _tempfile.mkstemp(dir=d, prefix=".notif_announce_sha256_",
                                    suffix=".tmp")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            _os.replace(tmp, _ANNOUNCE_SHA256_DEDUPE_FILE)
        except Exception:
            try: _os.remove(tmp)
            except Exception: pass
    except Exception:
        pass


_load_announce_sha256_dedup()


def _maybe_save_dedupe_state() -> None:
    """Save dedupe state on every change. The original 30s throttle meant
    a bounce inside the throttle window dropped fresh _seen_ids and let the
    listener re-announce the backlog. Atomic-rename writes are cheap; the
    FS hit per notification is acceptable to keep the persisted set in
    lockstep with memory across bounces."""
    _last_dedupe_save[0] = _time.time()
    _save_dedupe_state()
_listener_thread: list = [None]
_pause_flag           = [False]
_subsystem_status     = {
    "winsdk_available": False,
    "listener_access":  None,    # None / "Allowed" / "Denied" / "Unspecified"
    "listening":        False,
    "errors":           [],
    "started_at":       None,
    "last_poll_at":     None,
    "last_error":       None,
}


# ─── winsdk import dance ─────────────────────────────────────────────────
#
# winsdk re-exports the WinRT modules under several package layouts
# depending on version (winsdk 1.x vs winrt-runtime 2.x), and old code
# sometimes still imports `winrt.windows.*`. Probe each candidate path
# so the skill works against whatever the user actually has installed.

_winsdk_modules: dict[str, object] = {}


def _probe_winsdk() -> bool:
    """Try every known import path. Populates `_winsdk_modules` on
    success. Returns True if both UserNotificationListener and the
    notification-kind enum loaded."""
    candidates = [
        # (module_path, attr_name, key)
        ("winsdk.windows.ui.notifications.management",
         "UserNotificationListener", "Listener"),
        ("winsdk.windows.ui.notifications",
         "NotificationKinds", "NotificationKinds"),
        ("winsdk.windows.ui.notifications",
         "KnownNotificationBindings", "KnownNotificationBindings"),
    ]
    # Fallbacks for older winrt-runtime exposing the same surface under
    # the `winrt` namespace.
    fallback_pairs = [
        ("winrt.windows.ui.notifications.management",
         "UserNotificationListener", "Listener"),
        ("winrt.windows.ui.notifications",
         "NotificationKinds", "NotificationKinds"),
        ("winrt.windows.ui.notifications",
         "KnownNotificationBindings", "KnownNotificationBindings"),
    ]

    def _try(pairs):
        ok = True
        for mod_path, attr, key in pairs:
            try:
                m = importlib.import_module(mod_path)
                obj = getattr(m, attr)
                _winsdk_modules[key] = obj
            except Exception:
                ok = False
                break
        return ok

    if _try(candidates):
        return True
    _winsdk_modules.clear()
    return _try(fallback_pairs)


# ─── Rule storage ────────────────────────────────────────────────────────

def _load_rules() -> list[dict]:
    """Read notification_rules.json. Seed with DEFAULT_RULES on first
    boot. Returns an in-memory copy of the rule list, sorted by priority
    descending so the first match wins on the hot path."""
    if not os.path.exists(_RULES_FILE):
        try:
            _atomic_write_json(_RULES_FILE, DEFAULT_RULES)
        except Exception as e:
            _log.warning("[triage] could not seed default rules: %s", e)
        return _sort_rules(list(DEFAULT_RULES))
    try:
        with open(_RULES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cleaned = [r for r in data if _validate_rule(r)]
            return _sort_rules(cleaned)
    except Exception as e:
        _log.warning("[triage] rules file unreadable, falling back to defaults: %s", e)
    return _sort_rules(list(DEFAULT_RULES))


def _save_rules() -> None:
    with _state_lock:
        snapshot = [dict(r) for r in _rules]
    try:
        _atomic_write_json(_RULES_FILE, snapshot)
    except Exception as e:
        _log.warning("[triage] rules persist failed: %s", e)


def _validate_rule(r: object) -> bool:
    """Light schema check. Silently drop invalid entries rather than
    crashing the whole skill."""
    if not isinstance(r, dict):
        return False
    if not r.get("id") or not r.get("action"):
        return False
    if r["action"] not in ("read_aloud", "log", "drop", "classify"):
        return False
    if "match" in r and not isinstance(r["match"], dict):
        return False
    return True


def _sort_rules(rules: list[dict]) -> list[dict]:
    return sorted(rules, key=lambda r: -int(r.get("priority", 0)))


# ─── Rule evaluation ─────────────────────────────────────────────────────

def _rule_matches(rule: dict, app: str, title: str, body: str) -> bool:
    match = rule.get("match") or {}
    if not match:
        # Empty match = catch-all. Treat as "matches" so users can write a
        # final fallback rule with priority 0.
        return True
    for field, value in (("app_pattern", app), ("title_pattern", title),
                         ("body_pattern", body)):
        pattern = match.get(field)
        if pattern is None:
            continue
        try:
            if not re.search(pattern, value or ""):
                return False
        except re.error as e:
            _log.warning("[triage] rule %s has invalid regex on %s: %s",
                         rule.get("id"), field, e)
            return False
    return True


def _select_action(app: str, title: str, body: str) -> tuple[str, dict | None]:
    """Walk the rule list in priority order, return (action, rule). If
    no rule matches, defer to `classify` with rule=None so the caller can
    decide whether to invoke the LLM."""
    with _state_lock:
        rules_snapshot = list(_rules)
    for rule in rules_snapshot:
        if _rule_matches(rule, app, title, body):
            return rule.get("action", "classify"), rule
    return "classify", None


# ─── Haiku triage classifier ─────────────────────────────────────────────

def _classify_with_llm(app: str, title: str, body: str) -> str | None:
    """Returns one of {urgent, fyi, newsletter, spam} or None on error.
    Side-effect free; safe to call from the listener thread."""
    if not ENABLE_LLM_CLASSIFIER:
        return None

    prompt = (
        "You triage Windows desktop notifications for a personal assistant. "
        "Classify the toast into EXACTLY one of these labels (no extra text):\n"
        "  urgent     — needs the user's attention now (DM from a person, "
        "meeting starting, alert from a tool the user cares about, "
        "build/deploy/CI failure, security/2FA prompt).\n"
        "  fyi        — informational, useful to log but not interrupt "
        "(channel mentions, automated reports, status updates).\n"
        "  newsletter — bulk email or recurring digest the user opted into.\n"
        "  spam       — marketing, generic OS/Store nags, achievement "
        "popups, advertising.\n\n"
        f"App: {app or '(unknown)'}\n"
        f"Title: {title or '(empty)'}\n"
        f"Body: {body[:400] or '(empty)'}\n\n"
        "Answer with only the single label, lowercase."
    )
    _system = ("You are a precise notification triage classifier. "
               "Respond with one of: urgent, fyi, newsletter, spam.")

    def _parse_verdict(text: str) -> str | None:
        verdict = text.strip().lower().split()[0] if text.strip() else ""
        verdict = re.sub(r"[^a-z]", "", verdict)
        return verdict if verdict in LLM_VERDICTS_TO_ACTION else None

    # Primary path: Haiku/Claude (keeps cost low). Only attempted when an API
    # key is present and the SDK imports.
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=LLM_MODEL,
                max_tokens=8,
                system=_system,
                messages=[{"role": "user", "content": prompt}],
                timeout=LLM_TIMEOUT_SECONDS,
            )
            text = ""
            for block in getattr(msg, "content", []) or []:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    text += t
            verdict = _parse_verdict(text)
            if verdict:
                return verdict
        except Exception as e:
            _log.debug("[triage] Haiku classifier failed, trying local: %s", e)

    # Fallback path: local Ollama model (used while the Claude API is capped
    # or whenever the primary path raises / has no key). Parsed identically.
    try:
        bc = sys.modules.get("bobert_companion") or importlib.import_module("bobert_companion")
        local = bc._call_local_llm(
            _system, [{"role": "user", "content": prompt}], max_tokens=8)
        if local:
            verdict = _parse_verdict(local)
            if verdict:
                return verdict
    except Exception as e:
        _log.debug("[triage] local classifier failed: %s", e)
    return None


# ─── Speech queue + focus-mode integration ───────────────────────────────

def _proactive_announce(message: str) -> None:
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="notification_triage")
            return
    except Exception:
        pass
    print(f"  [triage] (no announcer) message: {message}")


def _focus_mode_active() -> bool:
    """Best-effort check — if the skill isn't loaded, always returns False."""
    mod = sys.modules.get("skill_dnd_focus_mode")
    if mod is None:
        return False
    try:
        fn = getattr(mod, "is_focus_mode_active", None)
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


# ─── Notification capture ────────────────────────────────────────────────

def _format_for_speech(app: str, title: str, body: str) -> str:
    """Compose a short, JARVIS-voice sentence. Examples:
      'Sir, Teams notification from Sam: are you around?'
      'Sir, Outlook reminder: standup in five minutes.'"""
    if not (title or body):
        return f"Sir, a notification from {app or 'your system'}."
    head = app.strip() if app else ""
    pieces = []
    if title:
        pieces.append(title.strip())
    if body and body.strip() and body.strip() != (title or "").strip():
        snippet = body.strip()
        if len(snippet) > MAX_SPOKEN_BODY_CHARS:
            snippet = snippet[: MAX_SPOKEN_BODY_CHARS - 1].rstrip() + "…"
        pieces.append(snippet)
    payload = " — ".join(pieces)
    if head:
        return f"Sir, {head} notification: {payload}"
    return f"Sir, notification: {payload}"


def _persist_to_log(record: dict) -> None:
    """Append a single notification record to today's jsonl. Cheap and
    crash-safe — append-mode + utf-8."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        day = time.strftime("%Y-%m-%d", time.localtime(record.get("ts", time.time())))
        path = os.path.join(_DATA_DIR, f"{day}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        _log.debug("[triage] log write failed: %s", e)


def _handle_notification(notification) -> None:
    """Process one UserNotification object from the listener queue."""
    try:
        nid = int(getattr(notification, "id", 0))
    except Exception:
        nid = 0
    if nid:
        with _state_lock:
            if nid in _seen_ids:
                return
            _seen_ids.add(nid)
            # Keep the seen set bounded — Windows reuses ids slowly so
            # 4000 is plenty of headroom but stops unbounded growth.
            if len(_seen_ids) > 4000:
                # Drop the oldest 1000 ids — set has no ordering so this
                # is approximate, but acceptable for dedupe purposes.
                for x in list(_seen_ids)[:1000]:
                    _seen_ids.discard(x)
        # task-75: persist so the next bounce doesn't see a fresh empty
        # set and re-announce. Persisted on every change (the old 30s throttle
        # was intentionally dropped — see _maybe_save_dedupe_state's docstring).
        _maybe_save_dedupe_state()

    app = ""
    title = ""
    body = ""
    try:
        info = getattr(notification, "app_info", None)
        if info is not None:
            disp = getattr(info, "display_info", None)
            if disp is not None:
                app = getattr(disp, "display_name", "") or ""
    except Exception:
        pass

    try:
        toast = notification.notification
        visual = toast.visual
        bindings = visual.bindings
        for binding in bindings:
            for txt in binding.get_text_elements():
                t = getattr(txt, "text", "") or ""
                if not title:
                    title = t
                elif not body:
                    body = t
                else:
                    body += " " + t
    except Exception as e:
        _log.debug("[triage] text extraction failed: %s", e)

    title = (title or "").strip()
    body = (body or "").strip()
    app = (app or "").strip()

    # Timestamp-bucket dedupe — catches rapid Teams/Windows retries where
    # the same (sender, body) arrives multiple times within a minute.
    # Runs before the broader 4-hour content_dedupe so the targeted Teams
    # case doesn't depend on app-name normalization (Teams flips between
    # "Microsoft Teams" / "Teams classic" / "Teams (work or school)").
    # Only engages when we actually have content; an all-empty toast
    # can't be meaningfully deduped.
    if title or body:
        now_ts = _time.time()
        _prune_notification_timestamp_dedup(now_ts)
        bucket = _timestamp_bucket(now_ts)
        ts_key = _notification_timestamp_dedupe_key(title, body, bucket)
        with _state_lock:
            prev_ts = _notification_timestamp_dedup.get(ts_key, 0.0)
            already_within_window = (
                prev_ts > 0.0
                and (now_ts - prev_ts) < _NOTIF_TS_DEDUPE_TTL_SEC
            )
            if already_within_window:
                return
            _notification_timestamp_dedup[ts_key] = now_ts
        _save_notification_timestamp_dedup()

    # Content-based dedupe — guards against WinRT-id regeneration after a
    # watchdog stall/restart, which the integer-id set above misses
    # because Windows hands out fresh ids for the same backlog toast.
    # Only engage when we actually have content; an all-empty toast can't
    # be meaningfully deduped and we'd rather let it through than collapse
    # unrelated empty notifications.
    if title or body:
        content_key = _content_dedupe_key(app, title, body)
        now_ts = _time.time()
        # Read-check-write the content dedupe in ONE critical section. The split
        # version (read under lock, RELEASE, test, re-acquire to write) let two
        # listener threads processing the same backlog toast -- WinRT hands out
        # fresh ids after a watchdog stall, so the id-set above misses it -- both
        # observe prev_ts as stale, both pass the TTL test, and both announce.
        # Holding _state_lock across get + test + set makes it atomic; the loser
        # sees the winner's write and suppresses. (return exits the with-block,
        # releasing the RLock.)
        with _state_lock:
            prev_ts = _content_dedupe.get(content_key, 0.0)
            if now_ts - prev_ts < _CONTENT_DEDUPE_TTL_SEC:
                return
            _content_dedupe[content_key] = now_ts
        _save_content_dedupe()

    action, rule = _select_action(app, title, body)
    verdict_label = None

    if action == "classify":
        verdict_label = _classify_with_llm(app, title, body)
        if verdict_label:
            action = LLM_VERDICTS_TO_ACTION[verdict_label]
        else:
            # LLM unavailable or failed — log silently so nothing is lost.
            action = "log"

    record = {
        "ts": time.time(),
        "app": app,
        "title": title,
        "body": body,
        "action": action,
        "rule_id": (rule or {}).get("id"),
        "llm_verdict": verdict_label,
    }

    with _state_lock:
        _recent.append(record)
        if len(_recent) > MAX_NOTIFICATION_LOG:
            del _recent[0:len(_recent) - MAX_NOTIFICATION_LOG]

    if action == "drop":
        return

    _persist_to_log(record)

    if action == "log":
        return

    if action == "read_aloud":
        priority = int((rule or {}).get("priority", 0))
        focus_blocks = _focus_mode_active() and priority < HIGH_PRIORITY_FLOOR
        if focus_blocks:
            return
        # Dedupe / snooze
        dedupe_key = f"{app}|{title[:80]}"
        now = time.time()
        with _state_lock:
            last = _snooze.get(dedupe_key, 0.0)
        if now - last < SNOOZE_SECONDS:
            return
        with _state_lock:
            _snooze[dedupe_key] = now
        # Persist the snooze stamp before we announce so a bounce mid-TTS
        # doesn't lose the dedupe key and replay the same toast on restart.
        _maybe_save_dedupe_state()
        # Announce-specific SHA-1 dedup gate: even if every upstream cache
        # let this through, suppress when the same (source, sender, body)
        # was announced inside the last 10 minutes.
        sender_norm = title.split(maxsplit=1)[0] if title else ""
        announce_key = _announce_dedup_key(app, sender_norm, body or title)
        _prune_announce_dedup(now)
        with _state_lock:
            prev_announce_ts = _announce_dedup.get(announce_key, 0.0)
            if (prev_announce_ts > 0.0
                    and (now - prev_announce_ts) < _ANNOUNCE_DEDUPE_TTL_SEC):
                return
            _announce_dedup[announce_key] = now
        _save_announce_dedup()
        # Final SHA-256 dedup gate keyed on (sender, body, window_id).
        # window_id is the WinRT notification id captured at handler entry
        # — the closest stable per-window identifier the UserNotification
        # API exposes. 30-min TTL is the longest of any dedup cache and
        # exists specifically to suppress the "every poll cycle" repeat
        # bug that the SHA-1 / MD5 caches were occasionally missing.
        sha256_key = _announce_sha256_dedupe_key(
            sender_norm, body or title, nid
        )
        _prune_announce_sha256_dedup(now)
        with _state_lock:
            prev_sha256_ts = _announce_sha256_dedup.get(sha256_key, 0.0)
            if (prev_sha256_ts > 0.0
                    and (now - prev_sha256_ts) < _ANNOUNCE_SHA256_DEDUPE_TTL_SEC):
                return
            _announce_sha256_dedup[sha256_key] = now
        _save_announce_sha256_dedup()
        message = _format_for_speech(app, title, body)
        _proactive_announce(message)


# ─── Background poll loop ────────────────────────────────────────────────

def _request_access(listener) -> str:
    """Wraps RequestAccessAsync and waits on the resulting IAsyncOperation.
    Returns the access-status name ('Allowed', 'Denied', 'Unspecified')."""
    try:
        op = listener.request_access_async()
    except AttributeError:
        # winsdk < 1.0 used PascalCase; try the camelCase too.
        op = listener.RequestAccessAsync()
    try:
        result = op.get()
    except Exception:
        # winsdk awaitable -> use the asyncio integration via .get_results()
        # after waiting. Fall through to a manual poll.
        try:
            # 2026-07-08: bound the manual poll with a deadline — an op that
            # never flips .completed would otherwise busy-wait forever.
            _deadline = time.monotonic() + ASYNC_OP_TIMEOUT_SECONDS
            while not op.completed:
                if time.monotonic() >= _deadline:
                    raise TimeoutError("RequestAccessAsync did not complete in "
                                       f"{ASYNC_OP_TIMEOUT_SECONDS:.0f}s")
                time.sleep(0.05)
            result = op.get_results()
        except Exception as e:
            raise RuntimeError(f"RequestAccessAsync wait failed: {e}") from e
    try:
        return result.name  # winsdk enum exposes .name
    except Exception:
        return str(result)


def _get_notifications(listener):
    NotificationKinds = _winsdk_modules.get("NotificationKinds")
    if NotificationKinds is None:
        raise RuntimeError("NotificationKinds enum not loaded")
    try:
        op = listener.get_notifications_async(NotificationKinds.TOAST)
    except AttributeError:
        op = listener.GetNotificationsAsync(NotificationKinds.TOAST)
    try:
        return op.get()
    except Exception:
        try:
            # 2026-07-08: bound the manual poll with a deadline so a wedged
            # WinRT op can't spin the triage daemon forever — the loop's except
            # recovers on the next poll cycle.
            _deadline = time.monotonic() + ASYNC_OP_TIMEOUT_SECONDS
            while not op.completed:
                if time.monotonic() >= _deadline:
                    raise TimeoutError("GetNotificationsAsync did not complete "
                                       f"in {ASYNC_OP_TIMEOUT_SECONDS:.0f}s")
                time.sleep(0.05)
            return op.get_results()
        except Exception as e:
            raise RuntimeError(f"GetNotificationsAsync wait failed: {e}") from e


def _listener_loop() -> None:
    """Background poll thread. Reconnects to the listener on transient
    errors and surfaces fatal errors via subsystem_status."""
    time.sleep(INITIAL_DELAY_SECONDS)

    if not _probe_winsdk():
        _subsystem_status["winsdk_available"] = False
        _subsystem_status["last_error"] = (
            "winsdk not installed — pip install winsdk (Windows only)"
        )
        print("  [triage] winsdk not available; notification capture offline.")
        return
    _subsystem_status["winsdk_available"] = True

    Listener = _winsdk_modules["Listener"]
    try:
        listener = Listener.current
    except Exception:
        try:
            listener = Listener.Current
        except Exception as e:
            _subsystem_status["last_error"] = f"could not obtain listener: {e}"
            print(f"  [triage] could not obtain UserNotificationListener: {e}")
            return

    try:
        access = _request_access(listener)
    except Exception as e:
        _subsystem_status["last_error"] = f"RequestAccessAsync failed: {e}"
        print(f"  [triage] RequestAccessAsync failed: {e}")
        return
    _subsystem_status["listener_access"] = access
    if access != "Allowed":
        # The user can still grant it via Settings → Privacy → Notifications.
        print(f"  [triage] notification access = {access}; capture disabled until granted.")
        return

    _subsystem_status["listening"] = True
    _subsystem_status["started_at"] = time.time()
    print("  [triage] UserNotificationListener active — toast capture live.")

    consecutive_errors = 0
    while True:
        if _pause_flag[0]:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        try:
            notifications = _get_notifications(listener)
            _subsystem_status["last_poll_at"] = time.time()
            consecutive_errors = 0
            for n in notifications:
                try:
                    _handle_notification(n)
                except Exception as e:
                    _log.debug("[triage] notification handler crashed: %s\n%s",
                               e, traceback.format_exc())
        except Exception as e:
            consecutive_errors += 1
            _subsystem_status["last_error"] = str(e)
            if consecutive_errors <= 3 or consecutive_errors % 20 == 0:
                _log.warning("[triage] poll iteration failed (%d): %s",
                             consecutive_errors, e)
            # Back off briefly on repeated failures — usually a transient
            # WinRT marshalling hiccup.
            if consecutive_errors > 5:
                time.sleep(min(60.0, POLL_INTERVAL_SECONDS * 5))
                continue
        time.sleep(POLL_INTERVAL_SECONDS)


def _start_listener() -> None:
    if _listener_thread[0] is not None and _listener_thread[0].is_alive():
        return
    t = threading.Thread(target=_listener_loop, name="notification-triage",
                         daemon=True)
    t.start()
    _listener_thread[0] = t


# ─── Public helper for other skills ──────────────────────────────────────

def recent_notifications(n: int = 10) -> list[dict]:
    """Snapshot of the last `n` triaged toasts. Returns a list of dicts
    (shape matches the records in data/notifications/<date>.jsonl)."""
    with _state_lock:
        return list(_recent[-max(1, int(n)):])


# ─── Action handlers ─────────────────────────────────────────────────────

def _format_rule_for_voice(r: dict) -> str:
    match = r.get("match") or {}
    bits = []
    if "app_pattern" in match:
        bits.append(f"app~{match['app_pattern']}")
    if "title_pattern" in match:
        bits.append(f"title~{match['title_pattern'][:40]}")
    if "body_pattern" in match:
        bits.append(f"body~{match['body_pattern'][:40]}")
    spec = ", ".join(bits) or "*"
    return f"{r.get('id')} [p={r.get('priority', 0)}] {r.get('action')} when {spec}"


def register(actions):

    def triage_status(_: str = "") -> str:
        with _state_lock:
            rule_count = len(_rules)
            recent_count = len(_recent)
        last_record = recent_notifications(1)
        bits = []
        if not _subsystem_status["winsdk_available"]:
            bits.append("winsdk not installed")
        else:
            bits.append(f"access={_subsystem_status['listener_access']}")
            bits.append("listening" if _subsystem_status["listening"] else "idle")
        bits.append(f"{rule_count} rules")
        bits.append(f"{recent_count} recent")
        if _focus_mode_active():
            bits.append("focus mode engaged")
        if _pause_flag[0]:
            bits.append("paused")
        if _subsystem_status["last_error"]:
            bits.append(f"last error: {_subsystem_status['last_error'][:100]}")
        if last_record:
            r = last_record[0]
            bits.append(
                f"last: {r.get('app') or '?'} → {r.get('action')}"
            )
        return "Notification triage — " + "; ".join(bits) + ", sir."

    def list_notification_rules(_: str = "") -> str:
        with _state_lock:
            snapshot = list(_rules)
        if not snapshot:
            return "No notification rules configured, sir."
        lines = [_format_rule_for_voice(r) for r in snapshot[:25]]
        more = "" if len(snapshot) <= 25 else f" (and {len(snapshot) - 25} more)"
        return "Active rules, sir:\n" + "\n".join(lines) + more

    def add_notification_rule(arg: str) -> str:
        if not arg.strip():
            return ("Pass the rule as JSON, sir — e.g. "
                    "{\"id\":\"slack_general\",\"match\":{\"app_pattern\":"
                    "\"(?i)slack\",\"title_pattern\":\"#general\"},"
                    "\"action\":\"drop\",\"priority\":25,\"reason\":\"too noisy\"}.")
        try:
            rule = json.loads(arg)
        except Exception as e:
            return f"Could not parse the rule, sir — JSON error: {e}"
        if not _validate_rule(rule):
            return ("That rule is missing required fields, sir — needs id and "
                    "action (read_aloud|log|drop|classify).")
        with _state_lock:
            # Replace by id if already present, otherwise append.
            global _rules
            others = [r for r in _rules if r.get("id") != rule["id"]]
            others.append(rule)
            _rules = _sort_rules(others)
        _save_rules()
        return f"Rule {rule['id']} saved, sir."

    def remove_notification_rule(arg: str) -> str:
        rid = arg.strip()
        if not rid:
            return "Tell me which rule id to remove, sir."
        with _state_lock:
            global _rules
            before = len(_rules)
            _rules = [r for r in _rules if r.get("id") != rid]
            removed = before - len(_rules)
        if removed:
            _save_rules()
            return f"Removed rule {rid}, sir."
        return f"No rule with id {rid}, sir."

    def recent_notifications_action(arg: str = "") -> str:
        try:
            n = int(arg.strip()) if arg.strip() else 5
        except Exception:
            n = 5
        n = max(1, min(20, n))
        snapshot = recent_notifications(n)
        if not snapshot:
            return "Nothing in the notification log yet, sir."
        lines = []
        for r in snapshot:
            ago = max(0, int(time.time() - float(r.get("ts", time.time()))))
            head = r.get("app") or "?"
            title = (r.get("title") or "").strip() or "(no title)"
            tag = r.get("action", "?")
            lines.append(f"{ago}s ago — {head} — {title[:80]} [{tag}]")
        return "Recent notifications, sir:\n" + "\n".join(lines)

    def pause_notification_triage(_: str = "") -> str:
        _pause_flag[0] = True
        return "Notification triage paused, sir."

    def resume_notification_triage(_: str = "") -> str:
        _pause_flag[0] = False
        if _listener_thread[0] is None or not _listener_thread[0].is_alive():
            _start_listener()
        return "Notification triage resumed, sir."

    actions["triage_status"]                = triage_status
    actions["notification_triage_status"]   = triage_status
    actions["list_notification_rules"]      = list_notification_rules
    actions["add_notification_rule"]        = add_notification_rule
    actions["remove_notification_rule"]     = remove_notification_rule
    # NOTE: action key intentionally distinct from the module-level
    # `recent_notifications()` helper above (which returns list[dict]) so
    # `from skills.notification_triage import recent_notifications` and
    # `ACTIONS["recent_notifications_summary"]` don't fight over the same name.
    actions["recent_notifications_summary"] = recent_notifications_action
    actions["list_recent_notifications"]    = recent_notifications_action
    actions["pause_notification_triage"]    = pause_notification_triage
    actions["resume_notification_triage"]   = resume_notification_triage

    # Load rules (creates default file on first boot) and kick off the
    # listener thread. The thread sleeps INITIAL_DELAY_SECONDS before
    # probing winsdk so a fresh boot isn't slowed down.
    global _rules
    _rules = _load_rules()
    _start_listener()

    print("  [triage] notification_triage ready — actions: triage_status, "
          "list_notification_rules, add_notification_rule, "
          "remove_notification_rule, recent_notifications_summary, "
          "pause_notification_triage, resume_notification_triage")
