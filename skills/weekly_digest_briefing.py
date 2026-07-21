"""
Weekly-digest briefing — surface pattern_learning's habit clusters as
once-per-week proactive offers.

Where skills/anticipation_briefing.py works off the daily aggregator snapshot
(broad windows + precise-clock predictions), this skill reads the WEEKLY
cluster snapshot computed by skills/pattern_learning.py.compute_weekly_digest()
— habits grouped by (day-of-week, hour-window, action) over a 4-week lookback,
the MCU-JARVIS shape ('Friday 8 PM → Netflix', 'weekday 8 AM → Teams check').

Two surfacing modes:

  • LEAD-IN — if today's day-of-week matches a cluster and the current time is
    within WEEKLY_DIGEST_LEAD_MINUTES of the cluster's hour band, voice the
    cluster's offer ('Sir, it's Friday — and around 8 PM you usually queue
    Netflix. Shall I bring it up?').

  • INSIDE-BAND — if we're inside the hour band itself, voice the same line
    without the lead-in framing.

Throttling:
  Once-per-week-per-cluster-key (week label = ISO Monday). State file lives
  alongside the other skill state files. WEEKLY_DIGEST_MAX_CARDS hard cap on
  offers surfaced per day so a quiet evening doesn't dump three suggestions
  back-to-back.

Hard gates (mirror anticipation_briefing — all must pass before firing):
  • bobert_companion._sleep_mode / _standby_mode → suppress
  • in a Teams / Zoom / Meet / Webex call (per window titles) → suppress
  • face_tracker reports gaze == "away" → suppress
  • engine disabled via WEEKLY_DIGEST_ENABLED → no thread

Actions registered:
  weekly_digest_now      — force-fire the next eligible cluster offer
                           (bypasses the once-per-week throttle, still
                           respects hard gates)
  weekly_digest_status   — short status report

Config knobs in bobert_companion.py (all defensive defaults via getattr):
  WEEKLY_DIGEST_ENABLED        (bool,  default True)
  WEEKLY_DIGEST_POLL_MINUTES   (int,   default 15,  clamped to [1, 60])
  WEEKLY_DIGEST_LEAD_MINUTES   (int,   default 30,  clamped to [1, 120])
  WEEKLY_DIGEST_CONFIDENCE_MIN (float, default 0.5, clamped to [0, 1])
  WEEKLY_DIGEST_MAX_CARDS      (int,   default 3,   clamped to [1, 10])
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import tempfile
import threading
import time

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# This module is a runtime-state WRITER; a private join here is how a
# staging-isolated sweep overwrote the LIVE smart-home catalog.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_STATE_FILE   = os.path.join(_DATA_DIR, "weekly_digest_briefing_state.json")

INITIAL_DELAY_SECONDS = 180        # let JARVIS finish booting + pattern_learning seed
DEFAULT_POLL_MINUTES  = 15
DEFAULT_LEAD_MINUTES  = 30
DEFAULT_CONFIDENCE    = 0.5
DEFAULT_MAX_CARDS     = 3

# Reuse the same call-window hints as anticipation_briefing so the suppression
# logic stays consistent across the two engines.
CALL_WINDOW_HINTS = (
    "meeting now",
    "meeting in ",
    " | microsoft teams meeting",
    "microsoft teams meeting |",
    "zoom meeting",
    "zoom - meeting",
    "webex meetings",
    "google meet -",
    "meet -",
    "discord call",
)

_state_lock = threading.Lock()


# ─── config ──────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    try:
        v = type(lo)(v)
    except Exception:
        return lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _read_config() -> dict:
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        bc = None
    enabled = bool(getattr(bc, "WEEKLY_DIGEST_ENABLED", True)) if bc else True
    poll    = int  (getattr(bc, "WEEKLY_DIGEST_POLL_MINUTES",   DEFAULT_POLL_MINUTES))  if bc else DEFAULT_POLL_MINUTES
    lead    = int  (getattr(bc, "WEEKLY_DIGEST_LEAD_MINUTES",   DEFAULT_LEAD_MINUTES))  if bc else DEFAULT_LEAD_MINUTES
    conf    = float(getattr(bc, "WEEKLY_DIGEST_CONFIDENCE_MIN", DEFAULT_CONFIDENCE))    if bc else DEFAULT_CONFIDENCE
    cards   = int  (getattr(bc, "WEEKLY_DIGEST_MAX_CARDS",      DEFAULT_MAX_CARDS))     if bc else DEFAULT_MAX_CARDS
    return {
        "enabled":    enabled,
        "poll_min":   _clamp(poll,  1,   60),
        "lead_min":   _clamp(lead,  1,  120),
        "conf_floor": _clamp(conf,  0.0, 1.0),
        "max_cards":  _clamp(cards, 1,   10),
    }


# ─── once-per-week throttle ──────────────────────────────────────────────

def _ensure_data_dir() -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
    except Exception:
        pass


def _week_label(now: float | None = None) -> str:
    """ISO YYYY-MM-DD of the Monday on or before today (local clock)."""
    import datetime as _dt
    lt = time.localtime(now if now is not None else time.time())
    d = _dt.date(lt.tm_year, lt.tm_mon, lt.tm_mday)
    monday = d - _dt.timedelta(days=d.weekday())
    return monday.isoformat()


def _load_state() -> dict:
    if not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    _ensure_data_dir()
    tmp: str | None = None
    fd: int = -1
    try:
        with _state_lock:
            fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1   # fdopen took ownership of the descriptor
                json.dump(state, f, indent=2)
            os.replace(tmp, _STATE_FILE)
            tmp = None   # rename succeeded; nothing left to clean up
    except Exception as e:
        print(f"  [weekly_digest_briefing] state save failed: {e}")
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except Exception:
                pass
        if tmp is not None:
            try:
                os.unlink(tmp)
            except Exception:
                pass


def _prune_state(state: dict) -> None:
    """Drop entries older than 12 weeks so the file doesn't grow without bound."""
    try:
        import datetime as _dt
        cutoff = _dt.date.fromisoformat(_week_label()) - _dt.timedelta(weeks=12)
        cutoff_str = cutoff.isoformat()
        for k in list(state.keys()):
            v = state.get(k)
            if isinstance(v, str) and v < cutoff_str:
                del state[k]
    except Exception:
        pass


# ─── digest loader ───────────────────────────────────────────────────────

def _load_digest() -> dict:
    """Return the most recent weekly digest from pattern_learning. Lazy import
    so the load order doesn't matter. Returns {} when nothing is available."""
    pl = sys.modules.get("skill_pattern_learning")
    if pl is None:
        try:
            pl = importlib.import_module("skill_pattern_learning")
        except Exception:
            pl = None
    if pl is None:
        return {}
    loader = getattr(pl, "load_latest_weekly_digest", None)
    if not callable(loader):
        return {}
    try:
        data = loader()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ─── hard gates (sleep/standby/in-call/away) ─────────────────────────────

def _is_sleep_or_standby() -> bool:
    bc = sys.modules.get("bobert_companion")
    if bc is None:
        return False
    try:
        if getattr(bc, "_sleep_mode")[0]:
            return True
        if getattr(bc, "_standby_mode")[0]:
            return True
    except Exception:
        return False
    return False


def _all_window_titles() -> list[str]:
    try:
        import pygetwindow as gw   # type: ignore
    except Exception:
        return []
    out: list[str] = []
    try:
        for w in gw.getAllWindows():
            t = getattr(w, "title", "") or ""
            if t.strip():
                out.append(t)
    except Exception:
        pass
    return out


def _is_in_call() -> bool:
    titles = _all_window_titles()
    if not titles:
        return False
    lowered = [t.lower() for t in titles]
    for hint in CALL_WINDOW_HINTS:
        for t in lowered:
            if hint in t:
                return True
    return False


def _user_at_desk() -> bool | None:
    mod = sys.modules.get("skill_face_tracker")
    if mod is None:
        return None
    snap_func = getattr(mod, "_snapshot_state", None)
    if snap_func is None:
        return None
    try:
        snap = snap_func()
    except Exception:
        return None
    if not snap.get("last_sample_at"):
        return None
    monitor = snap.get("current_monitor")
    if monitor in ("left", "right", "middle_or_top"):
        return True
    if monitor == "away":
        return False
    return None


# ─── eligibility selection ───────────────────────────────────────────────

def _eligible_clusters(digest: dict, cfg: dict) -> list[dict]:
    """Return the digest clusters whose moment is approaching or current,
    sorted by (in-band first, then by lead time ascending, then by
    confidence descending). Filtered by config confidence floor."""
    if not digest:
        return []
    clusters = digest.get("clusters") or []
    if not clusters:
        return []
    now = time.localtime()
    today_dow  = now.tm_wday
    cur_minute = now.tm_hour * 60 + now.tm_min
    lead_min   = cfg["lead_min"]
    conf_floor = cfg["conf_floor"]

    out: list[tuple[int, int, dict]] = []   # (in_band_first, lead_minutes, cluster)
    for c in clusters:
        if not isinstance(c, dict):
            continue
        if float(c.get("confidence", 0.0) or 0.0) < conf_floor:
            continue
        if int(c.get("dow", -1)) != today_dow:
            continue
        hour_start = int(c.get("hour_start", -1))
        if hour_start < 0:
            continue
        try:
            hour_end = int(c.get("hour_end", hour_start + 2))
        except Exception:
            hour_end = hour_start + 2
        if hour_end <= hour_start:
            # Wrap-past-midnight isn't expected for the 2h weekly bands, but
            # defend against it by forcing the canonical 2-hour width.
            hour_end = hour_start + 2
        band_start_min = hour_start * 60
        band_end_min   = hour_end   * 60

        in_band = band_start_min <= cur_minute < band_end_min
        lead = band_start_min - cur_minute    # positive if upcoming
        if in_band:
            out.append((0, 0, c))
            continue
        if 0 < lead <= lead_min:
            out.append((1, lead, c))

    out.sort(key=lambda x: (x[0], x[1], -float(x[2].get("confidence", 0.0) or 0.0)))
    return [c for _, _, c in out]


def _compose_line(cluster: dict) -> str:
    """Return the spoken offer for this cluster. Defers to the offer string
    already baked into the digest (computed in pattern_learning); falls back
    to a generic phrasing if absent."""
    offer = (cluster.get("offer") or "").strip()
    if offer:
        return offer
    label = (cluster.get("label") or "").strip()
    return f"Sir — {label}. Shall I proceed?" if label else ""


# ─── speech queue ────────────────────────────────────────────────────────

def _enqueue_speech(message: str) -> bool:
    """Route through bobert_companion.proactive_announce; fall back to a
    direct pending_speech.json write so the loop still works during boot
    races."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            return bool(announcer(message, source="weekly_digest_briefing"))
    except Exception:
        pass
    queue_path = os.path.join(_PROJECT_DIR, "pending_speech.json")
    try:
        data: list = []
        if os.path.exists(queue_path):
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or []
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        fd: int = -1
        tmp: str | None = None
        try:
            fd, tmp = tempfile.mkstemp(dir=_PROJECT_DIR, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1   # fdopen took ownership of the descriptor
                json.dump(data, f, indent=2)
            os.replace(tmp, queue_path)
            tmp = None
        except Exception:
            if fd >= 0:
                try: os.close(fd)
                except Exception: pass
            if tmp is not None:
                try: os.unlink(tmp)
                except Exception: pass
            raise
        return True
    except Exception as e:
        print(f"  [weekly_digest_briefing] speech-queue write failed ({e}); line: {message}")
        return False


# ─── one-shot pick (used by loop AND manual action) ─────────────────────

def _next_eligible(bypass_throttle: bool) -> tuple[str, dict]:
    cfg = _read_config()
    digest = _load_digest()
    if not digest:
        return "", {}
    candidates = _eligible_clusters(digest, cfg)
    if not candidates:
        return "", {}

    week = _week_label()
    state = _load_state() if not bypass_throttle else {}

    cards_today = 0
    today = time.strftime("%Y-%m-%d", time.localtime())
    if not bypass_throttle:
        cards_today = sum(
            1 for k, v in state.items()
            if isinstance(v, dict) and v.get("day") == today
        )
        if cards_today >= cfg["max_cards"]:
            return "", {}

    for c in candidates:
        key = (c.get("key") or "").strip()
        if not key:
            continue
        if not bypass_throttle:
            prev = state.get(key)
            if isinstance(prev, dict) and prev.get("week") == week:
                continue
            # Backward-compat: pre-2026-05 saves stored a bare string.
            if isinstance(prev, str) and prev == week:
                continue
        line = _compose_line(c)
        if not line:
            continue
        return line, c
    return "", {}


def _mark_fired(cluster: dict) -> None:
    key = (cluster.get("key") or "").strip()
    if not key:
        return
    state = _load_state()
    state[key] = {
        "week": _week_label(),
        "day":  time.strftime("%Y-%m-%d", time.localtime()),
        "ts":   time.time(),
    }
    _prune_state(state)
    _save_state(state)


# ─── background scheduler ────────────────────────────────────────────────

def _scheduler_loop() -> None:
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        cfg = _read_config()
        poll_seconds = max(60, cfg["poll_min"] * 60)
        try:
            if not cfg["enabled"]:
                time.sleep(poll_seconds)
                continue

            if _is_sleep_or_standby():
                time.sleep(poll_seconds)
                continue
            if _is_in_call():
                time.sleep(poll_seconds)
                continue
            if _user_at_desk() is False:
                time.sleep(poll_seconds)
                continue

            line, cluster = _next_eligible(bypass_throttle=False)
            if line:
                print(f"  [weekly_digest_briefing] firing: {line}")
                if _enqueue_speech(line):
                    _mark_fired(cluster)
        except Exception:
            logging.exception("  [weekly_digest_briefing] scheduler error")
        time.sleep(poll_seconds)


# ─── actions ─────────────────────────────────────────────────────────────

def _act_weekly_digest_now(_: str = "") -> str:
    """Force-fire the next eligible weekly-digest card, bypassing the
    once-per-week throttle. Still gated by sleep/standby/in-call/away."""
    if _is_sleep_or_standby():
        return "Suppressed, sir — sleep or standby mode is active."
    if _is_in_call():
        return "Suppressed, sir — you appear to be on a call."
    if _user_at_desk() is False:
        return "Suppressed, sir — you do not appear to be at the desk."
    line, cluster = _next_eligible(bypass_throttle=True)
    if not line:
        return "No weekly habit matches the current moment, sir."
    if _enqueue_speech(line):
        _mark_fired(cluster)
        return line
    return f"Could not enqueue speech, sir. Line was: {line}"


def _act_weekly_digest_status(_: str = "") -> str:
    cfg = _read_config()
    digest = _load_digest()
    clusters = (digest.get("clusters") or []) if digest else []
    candidates = _eligible_clusters(digest, cfg) if digest else []

    state = _load_state()
    today = time.strftime("%Y-%m-%d", time.localtime())
    fired_today = sum(
        1 for v in state.values()
        if isinstance(v, dict) and v.get("day") == today
    )

    parts: list[str] = []
    if not cfg["enabled"]:
        parts.append("disabled in config")
    else:
        parts.append(
            f"poll every {cfg['poll_min']} min, lead {cfg['lead_min']} min, "
            f"confidence ≥ {cfg['conf_floor']:.0%}, ≤{cfg['max_cards']} cards/day"
        )
    gen_at = float(digest.get("computed_at", 0.0) or 0.0) if digest else 0.0
    if gen_at:
        age_h = (time.time() - gen_at) / 3600.0
        parts.append(f"{len(clusters)} clusters (digest {age_h:.1f}h old)")
    else:
        parts.append("no weekly digest cached yet")
    parts.append(f"{len(candidates)} eligible right now")
    parts.append(f"{fired_today} card{'s' if fired_today != 1 else ''} surfaced today")
    if _is_sleep_or_standby():
        parts.append("sleep/standby active (suppressed)")
    if _is_in_call():
        parts.append("in a call (suppressed)")
    if _user_at_desk() is False:
        parts.append("user away (suppressed)")
    return "Weekly digest briefing — " + "; ".join(parts) + "."


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["weekly_digest_now"]    = _act_weekly_digest_now
    actions["weekly_digest_status"] = _act_weekly_digest_status

    _ensure_data_dir()
    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [weekly_digest_briefing] WEEKLY_DIGEST_ENABLED is False — scheduler disabled")
        return

    digest = _load_digest()
    if not digest:
        print("  [weekly_digest_briefing] WARN: no weekly digest cached yet "
              "— briefings will be silent until pattern_learning runs the Monday job")
    else:
        n_clusters = len(digest.get("clusters") or [])
        gen_at = float(digest.get("computed_at", 0.0) or 0.0)
        age_h = (time.time() - gen_at) / 3600.0 if gen_at else float("inf")
        print(f"  [weekly_digest_briefing] digest: {n_clusters} clusters "
              f"(computed {age_h:.1f}h ago)")
    try:
        bc = importlib.import_module("bobert_companion")
        if not callable(getattr(bc, "proactive_announce", None)):
            print("  [weekly_digest_briefing] WARN: bobert_companion.proactive_announce "
                  "is not callable — falling back to direct pending_speech.json writes")
    except Exception as e:
        print(f"  [weekly_digest_briefing] WARN: bobert_companion import failed ({e})")

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print(
        f"  [weekly_digest_briefing] background loop running "
        f"(poll {cfg['poll_min']}m, lead {cfg['lead_min']}m, "
        f"confidence ≥ {cfg['conf_floor']:.0%}, ≤{cfg['max_cards']} cards/day)"
    )


# ─── offline smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("config:", _read_config())
    d = _load_digest()
    print(f"digest week_start={d.get('week_start')!r} "
          f"clusters={len(d.get('clusters') or [])}")
    cands = _eligible_clusters(d, _read_config())
    print(f"eligible right now: {len(cands)}")
    for c in cands[:5]:
        print("  ", c.get("key"), "→", _compose_line(c))
    line, c = _next_eligible(bypass_throttle=True)
    print("next eligible (bypass throttle):", repr(line))
