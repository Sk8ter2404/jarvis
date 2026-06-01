"""
Anticipation briefing — daily proactive surfacing of pattern_learning predictions.

Where skills/anticipation_engine.py picks ONE in-character line based on dwell
or pattern_memory, this skill works directly off the pattern_learning snapshot
(skills/pattern_learning.py → data/usage_patterns_aggregated.json) and turns
the broad/precise predictions into anticipatory speech at relevant times.

Two surfacing modes:
  • Precise-clock predictions (e.g. "the user checks Teams at 09:15 ± 5 min daily")
    are surfaced with LEAD TIME — when the predicted moment is within
    ANTICIPATION_BRIEFING_LEAD_MINUTES of the current clock, JARVIS volunteers a
    line like "Sam sync in 12 minutes, sir — shall I pull up your last
    conversation?". Composed from the action verb plus the prediction's
    common_arg when available.

  • Broad-window predictions (e.g. "the user plays music between 9-11am on
    weekdays 78% of the time") fire when the current hour falls inside the
    window — "Shall I queue your usual, sir? Michael Jackson, by the look of
    recent patterns." Tailored to common_arg when present.

Throttling:
  Independent once-per-day-per-prediction-key state file so it doesn't
  interfere with pattern_memory or pattern_learning's own throttle. Stale
  entries older than 90 days are pruned on save.

Hard gates (must all pass before firing — mirror anticipation_engine):
  • bobert_companion._sleep_mode / _standby_mode → suppress
  • in a Teams / Zoom / Meet / Webex call (per window titles) → suppress
  • face_tracker reports gaze == "away" → suppress
  • engine disabled via ANTICIPATION_BRIEFING_ENABLED → no thread

Actions registered:
  anticipation_briefing_now    — force-fire the next eligible briefing
                                  (bypasses the once-per-day throttle but
                                  still respects hard gates)
  anticipation_briefing_status — short status report

Config knobs in bobert_companion.py (all defensive defaults via getattr):
  ANTICIPATION_BRIEFING_ENABLED        (bool,  default True)
  ANTICIPATION_BRIEFING_POLL_MINUTES   (int,   default 5,   clamped to [1, 60])
  ANTICIPATION_BRIEFING_LEAD_MINUTES   (int,   default 15,  clamped to [1, 60])
  ANTICIPATION_BRIEFING_CONFIDENCE_MIN (float, default 0.5, clamped to [0, 1])
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
_DATA_DIR     = os.path.join(_PROJECT_DIR, "data")
_STATE_FILE   = os.path.join(_DATA_DIR, "anticipation_briefing_state.json")

INITIAL_DELAY_SECONDS = 120        # let JARVIS finish booting
DEFAULT_POLL_MINUTES  = 5
DEFAULT_LEAD_MINUTES  = 15
DEFAULT_CONFIDENCE    = 0.5

# Window-title fragments indicating an active call — reused from
# anticipation_engine so we suppress consistently.
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

_state_lock  = threading.Lock()


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
    enabled = bool(getattr(bc, "ANTICIPATION_BRIEFING_ENABLED", True)) if bc else True
    poll    = int  (getattr(bc, "ANTICIPATION_BRIEFING_POLL_MINUTES",   DEFAULT_POLL_MINUTES))  if bc else DEFAULT_POLL_MINUTES
    lead    = int  (getattr(bc, "ANTICIPATION_BRIEFING_LEAD_MINUTES",   DEFAULT_LEAD_MINUTES))  if bc else DEFAULT_LEAD_MINUTES
    conf    = float(getattr(bc, "ANTICIPATION_BRIEFING_CONFIDENCE_MIN", DEFAULT_CONFIDENCE))    if bc else DEFAULT_CONFIDENCE
    return {
        "enabled":    enabled,
        "poll_min":   _clamp(poll, 1, 60),
        "lead_min":   _clamp(lead, 1, 60),
        "conf_floor": _clamp(conf, 0.0, 1.0),
    }


# ─── persistent state (once-per-day per-key throttle) ────────────────────

def _ensure_data_dir() -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
    except Exception:
        pass


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
    with _state_lock:
        try:
            fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp, _STATE_FILE)
            except Exception:
                try: os.unlink(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  [anticipation_briefing] state save failed: {e}")


def _prune_state(state: dict) -> None:
    """Drop entries older than 90 days so the file doesn't grow without bound."""
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


# ─── pattern_learning snapshot reader ────────────────────────────────────

def _load_snapshot() -> dict:
    """Read the latest pattern_learning aggregated snapshot. Returns {} when
    the file is missing or unparseable. Lazy-importing here means we don't
    explode at register-time if pattern_learning hasn't loaded yet."""
    try:
        pl = sys.modules.get("skill_pattern_learning")
        if pl is None:
            pl = importlib.import_module("skill_pattern_learning")
    except Exception:
        pl = None
    if pl is not None:
        try:
            data = pl._load_aggregated()
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    # Fallback: read the file directly using the known path.
    path = os.path.join(_DATA_DIR, "usage_patterns_aggregated.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _bucket_for_weekday(wd: int) -> str:
    return "weekday" if 0 <= wd <= 4 else "weekend"


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
    """True if face_tracker recently saw the user, False if 'away', None if
    tracker isn't loaded / has no data."""
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


# ─── selecting eligible predictions for the current moment ───────────────

def _minutes_until(center_minute: int, cur_minute: int) -> int:
    """Forward-only minute delta from cur to center on the same day. Returns
    a non-negative value or a large sentinel if center has already passed
    today (we only surface predictions BEFORE they happen)."""
    delta = center_minute - cur_minute
    return delta if delta >= 0 else 10_000


def _select_predictions(snapshot: dict, cfg: dict) -> list[dict]:
    """Return predictions whose moment is approaching or current, filtered by
    confidence floor and sorted: precise-with-lead first, then broad-now."""
    if not snapshot:
        return []
    now = time.localtime()
    wd  = now.tm_wday
    cur_bucket = _bucket_for_weekday(wd)
    cur_minute = now.tm_hour * 60 + now.tm_min
    lead_min   = cfg["lead_min"]
    conf       = cfg["conf_floor"]

    candidates: list[tuple[int, dict]] = []   # (sort_key, prediction-with-meta)

    # Precise-clock: surface when within lead_min of the predicted center.
    for p in snapshot.get("precise", []) or []:
        if float(p.get("ratio", 0.0) or 0.0) < conf:
            continue
        center = p.get("center_minute")
        if not isinstance(center, int):
            continue
        tol = int(p.get("tolerance_min", 0) or 0)
        delta = _minutes_until(center, cur_minute)
        # Eligible if we're inside the lead window OR we're already inside the
        # tolerance band (i.e. the event is "happening now"). Skip if delta is
        # the past-event sentinel and we're outside the tolerance band.
        within_lead = 0 <= delta <= lead_min
        within_band = abs(cur_minute - center) <= tol
        if not (within_lead or within_band):
            continue
        meta = dict(p)
        meta["__lead_minutes"]  = max(delta, 0) if within_lead else 0
        meta["__sort_priority"] = 0     # precise wins ties
        candidates.append((meta["__lead_minutes"], meta))

    # Broad-window: surface when current hour is inside the window AND the
    # bucket matches today's day-of-week.
    for p in snapshot.get("broad", []) or []:
        if float(p.get("ratio", 0.0) or 0.0) < conf:
            continue
        if p.get("bucket") != cur_bucket:
            continue
        win = p.get("hour_window") or []
        if len(win) != 2:
            continue
        try:
            ws, we = int(win[0]), int(win[1])
        except (TypeError, ValueError):
            continue
        if not (ws <= now.tm_hour < we):
            continue
        meta = dict(p)
        meta["__lead_minutes"]  = 0
        meta["__sort_priority"] = 1
        # Sort broad after precise; among broads, higher-ratio first.
        candidates.append((10_000 - int(float(meta.get("ratio", 0)) * 1000), meta))

    # Final ordering: precise (priority 0) before broad (priority 1); within
    # precise the soonest event wins; within broad highest ratio wins.
    candidates.sort(key=lambda x: (x[1]["__sort_priority"], x[0]))
    return [c[1] for c in candidates]


# ─── composing the spoken line ───────────────────────────────────────────

def _titlecase(s: str) -> str:
    small = {"and", "or", "of", "the", "in", "on", "for", "to", "a", "an"}
    parts = (s or "").split()
    out = []
    for i, w in enumerate(parts):
        if i > 0 and w in small:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _compose_briefing_line(p: dict) -> str:
    action = (p.get("action") or "").strip()
    arg    = (p.get("common_arg") or "").strip()
    lead   = int(p.get("__lead_minutes", 0) or 0)
    fallback_offer = (p.get("offer") or "").strip()

    # Spec examples:
    #   "Shall I queue your usual, sir? Your favourite artist, by the look of recent patterns."
    #   "Sam sync in 12 minutes, sir — shall I pull up your last conversation?"
    if action in ("play_music", "resume_music", "youtube_play",
                  "spotify", "apple_music"):
        if arg:
            artist = _titlecase(arg)
            return (
                f"Shall I queue your usual, sir? "
                f"{artist}, by the look of recent patterns."
            )
        return "Shall I queue your usual playlist, sir? It is about that time."

    if action == "check_teams":
        # Contact name from common_arg gives the spec's headline line.
        if arg and lead > 0:
            who = _titlecase(arg)
            return (
                f"{who} sync in {lead} minute{'s' if lead != 1 else ''}, sir "
                f"— shall I pull up your last conversation?"
            )
        if arg:
            who = _titlecase(arg)
            return (
                f"Your {who} sync is upon us, sir "
                f"— shall I pull up your last conversation?"
            )
        if lead > 0:
            return (
                f"Teams sync in {lead} minute{'s' if lead != 1 else ''}, sir "
                f"— shall I pull up your most recent thread?"
            )
        return "You usually check Teams about now, sir — shall I take a look?"

    if action == "morning_briefing":
        if lead > 0:
            return (
                f"Morning briefing usually runs in about {lead} "
                f"minute{'s' if lead != 1 else ''}, sir — shall I deliver it?"
            )
        return "Shall I deliver the morning briefing, sir?"

    if action == "evening_briefing":
        if lead > 0:
            return (
                f"Evening briefing usually runs in about {lead} "
                f"minute{'s' if lead != 1 else ''}, sir — shall I deliver it?"
            )
        return "Shall I deliver the evening briefing, sir?"

    # Generic fall-through: lead-aware prefix plus the pattern_learning
    # `offer` field (already a complete JARVIS-style sentence).
    if lead > 0 and fallback_offer:
        return (
            f"In about {lead} minute{'s' if lead != 1 else ''}, sir — "
            f"{fallback_offer}"
        )
    return fallback_offer or ""


# ─── speech queue (routed through bobert_companion.proactive_announce) ───

def _enqueue_speech(message: str) -> bool:
    """Route the briefing through bobert_companion.proactive_announce so we
    inherit its atomic-write semantics and `source` tagging. Returns True on
    successful enqueue."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            return bool(announcer(message, source="anticipation_briefing"))
    except Exception:
        pass
    # Fall-through: best-effort direct write so a missing parent module
    # doesn't silence the briefing entirely.
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
        fd, tmp = tempfile.mkstemp(dir=_PROJECT_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, queue_path)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
            raise
        return True
    except Exception as e:
        print(f"  [anticipation_briefing] speech-queue write failed ({e}); line: {message}")
        return False


# ─── one-shot pick (used by the loop AND the manual action) ──────────────

def _next_eligible(bypass_throttle: bool) -> tuple[str, dict]:
    """Return (line, prediction) for the next eligible briefing, or
    ('', {}) if nothing matches now. Honors the once-per-day throttle unless
    bypass_throttle=True."""
    cfg = _read_config()
    snapshot = _load_snapshot()
    if not snapshot:
        return "", {}
    candidates = _select_predictions(snapshot, cfg)
    if not candidates:
        return "", {}

    today = time.strftime("%Y-%m-%d", time.localtime())
    state = _load_state() if not bypass_throttle else {}

    for p in candidates:
        key = (p.get("key") or "").strip()
        if not key:
            continue
        if not bypass_throttle and state.get(key) == today:
            continue
        line = _compose_briefing_line(p)
        if not line:
            continue
        return line, p
    return "", {}


def _mark_fired(pred: dict) -> None:
    key = (pred.get("key") or "").strip()
    if not key:
        return
    today = time.strftime("%Y-%m-%d", time.localtime())
    state = _load_state()
    state[key] = today
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

            # Hard gates — silently skip this poll.
            if _is_sleep_or_standby():
                time.sleep(poll_seconds)
                continue
            if _is_in_call():
                time.sleep(poll_seconds)
                continue
            if _user_at_desk() is False:
                time.sleep(poll_seconds)
                continue

            line, pred = _next_eligible(bypass_throttle=False)
            if line:
                print(f"  [anticipation_briefing] firing: {line}")
                if _enqueue_speech(line):
                    _mark_fired(pred)
        except Exception:
            logging.exception("  [anticipation_briefing] scheduler error")
        time.sleep(poll_seconds)


# ─── actions ─────────────────────────────────────────────────────────────

def _act_anticipation_briefing_now(_: str = "") -> str:
    """Force-fire the next eligible briefing, bypassing the once-per-day
    throttle. Still gated by sleep/standby/in-call/away so we don't bark
    at an empty desk or a live meeting."""
    if _is_sleep_or_standby():
        return "Suppressed, sir — sleep or standby mode is active."
    if _is_in_call():
        return "Suppressed, sir — you appear to be on a call."
    if _user_at_desk() is False:
        return "Suppressed, sir — you do not appear to be at the desk."
    line, pred = _next_eligible(bypass_throttle=True)
    if not line:
        return "No prediction matches the current moment, sir."
    if _enqueue_speech(line):
        # Even on a bypass-throttle fire we record it so the regular loop
        # doesn't replay the same line minutes later.
        _mark_fired(pred)
        return line
    return f"Could not enqueue speech, sir. Line was: {line}"


def _act_anticipation_briefing_status(_: str = "") -> str:
    cfg = _read_config()
    snapshot = _load_snapshot()
    n_broad   = len(snapshot.get("broad",   []) or []) if snapshot else 0
    n_precise = len(snapshot.get("precise", []) or []) if snapshot else 0
    candidates = _select_predictions(snapshot, cfg) if snapshot else []

    state   = _load_state()
    today   = time.strftime("%Y-%m-%d", time.localtime())
    fired   = sum(1 for v in state.values() if v == today)

    parts: list[str] = []
    if not cfg["enabled"]:
        parts.append("disabled in config")
    else:
        parts.append(
            f"poll every {cfg['poll_min']} min, lead {cfg['lead_min']} min, "
            f"confidence ≥ {cfg['conf_floor']:.0%}"
        )
    parts.append(f"{n_broad} broad, {n_precise} precise patterns in snapshot")
    parts.append(f"{len(candidates)} eligible right now")
    parts.append(f"{fired} briefing{'s' if fired != 1 else ''} surfaced today")
    if _is_sleep_or_standby():
        parts.append("sleep/standby active (suppressed)")
    if _is_in_call():
        parts.append("in a call (suppressed)")
    if _user_at_desk() is False:
        parts.append("user away (suppressed)")
    return "Anticipation briefing — " + "; ".join(parts) + "."


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["anticipation_briefing_now"]    = _act_anticipation_briefing_now
    actions["anticipation_briefing_status"] = _act_anticipation_briefing_status

    _ensure_data_dir()
    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [anticipation_briefing] ANTICIPATION_BRIEFING_ENABLED is False — scheduler disabled")
        return

    # Startup health-check — snapshot freshness and announcer wiring. Surfaces
    # silent-failure modes the planner flagged (first-boot empty snapshot,
    # missing proactive_announce) at register time so they're visible in the
    # boot log rather than hidden behind a quiet scheduler.
    snap = _load_snapshot()
    if not snap:
        print("  [anticipation_briefing] WARN: no pattern_learning snapshot yet "
              "— briefings will be silent until aggregation runs")
    else:
        n_broad   = len(snap.get("broad",   []) or [])
        n_precise = len(snap.get("precise", []) or [])
        gen_at    = float(snap.get("generated_at", 0.0) or 0.0)
        age_h     = (time.time() - gen_at) / 3600.0 if gen_at else float("inf")
        print(f"  [anticipation_briefing] snapshot: {n_broad} broad, "
              f"{n_precise} precise (aggregated {age_h:.1f}h ago)")
    try:
        bc = importlib.import_module("bobert_companion")
        if not callable(getattr(bc, "proactive_announce", None)):
            print("  [anticipation_briefing] WARN: bobert_companion.proactive_announce "
                  "is not callable — falling back to direct pending_speech.json writes")
    except Exception as e:
        print(f"  [anticipation_briefing] WARN: bobert_companion import failed ({e}) "
              "— direct speech-queue fallback will be used")

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print(
        f"  [anticipation_briefing] background loop running "
        f"(poll {cfg['poll_min']}m, lead {cfg['lead_min']}m, "
        f"confidence ≥ {cfg['conf_floor']:.0%})"
    )


# ─── offline smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("Reading snapshot:", os.path.join(_DATA_DIR, "usage_patterns_aggregated.json"))
    cfg = _read_config()
    print("config:", cfg)
    snap = _load_snapshot()
    print(f"snapshot: {len(snap.get('broad', []) or [])} broad, "
          f"{len(snap.get('precise', []) or [])} precise")
    cands = _select_predictions(snap, cfg)
    print(f"candidates now: {len(cands)}")
    for c in cands[:5]:
        print("  ", c.get("type"), c.get("key"), "→",
              _compose_briefing_line(c))
    line, pred = _next_eligible(bypass_throttle=True)
    print("next eligible (bypass throttle):", repr(line))
