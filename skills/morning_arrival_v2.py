"""
morning_arrival_v2 — presence-triggered morning briefing for JARVIS.

Companion to skills/morning_arrival.py. Where v1 fires on the first wake
event of the day (via skills/morning_chain.py), v2 gates on actual face
presence: it waits until the face_tracker confirms the user has been in
frame for SUSTAINED_PRESENCE_SECONDS, then delivers ONE 60-second briefing
chaining six overnight-relevant sources:

  1. weather_briefing.get_umbrella_alert("today")
  2. ms_graph.get_teams_unread_count()      (unread Teams chats + top sender)
  3. ms_graph.get_first_meeting("today")    (today's calendar)
  4. bambu_monitor — overnight print completion / current state
  5. amazon_order_tracker.action_check_orders()
  6. news_briefing.get_news_text()          (top 3 headlines)

Each section is gathered in parallel with a per-section hard timeout
(SECTION_TIMEOUT_SECONDS); a hung Graph call can't stall the briefing. The
composed text is then trimmed to fit TTS_BUDGET_SECONDS by progressively
dropping low-priority sections in TTS_DROP_ORDER.

Trigger paths:
  • Background daemon thread polls skills.face_tracker._state for sustained
    face-visible frames inside ARRIVAL_START_HOUR..ARRIVAL_END_HOUR local.
  • Public action "morning_arrival_v2" (manual force=True).
  • _fire_from_chain(reason) — entry for skills/morning_chain.py once
    extended to know about the v2 variant.

Same-day suppression is persisted to morning_arrival_v2_state.json via
core.atomic_io._atomic_write_json so a JARVIS restart can't re-fire and
two concurrent triggers can't double-fire.

Failure of any single source degrades the briefing (that section is left
out) rather than crashing it. Failure of face_tracker (no _state dict at
all) is a hard skip — the briefing never fires until presence is observable.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_STATE_FILE   = os.path.join(_PROJECT_DIR, "morning_arrival_v2_state.json")
_BAMBU_STATE  = os.path.join(_PROJECT_DIR, "bambu_overlay_state.json")

# Morning window. Outside this window, sustained presence does not trigger
# the briefing — the user walking past the cameras at 3 PM is not an arrival.
ARRIVAL_START_HOUR = 6
ARRIVAL_END_HOUR   = 12

# Minimum continuous face-visible duration before we count presence as a
# real "arrival" rather than someone briefly passing through frame. Matches
# the planner's ≥3 s hysteresis spec and aligns with face_tracker's own
# FACE_FRESH_SECONDS tolerance.
SUSTAINED_PRESENCE_SECONDS = 3.0

# Poll cadence for the presence watcher. The face_tracker poller runs at
# 0.5 s — we sample slightly faster than its sustained-presence threshold
# so the rising edge is caught within ~1 s of being eligible.
WATCH_POLL_SECONDS = 1.0

# Per-source timeout. Each of the six sections runs in its own thread and
# any section that overruns is dropped from the brief. News summarisation
# does a per-headline LLM call, so 8s leaves headroom for ~3 hops.
SECTION_TIMEOUT_SECONDS = 8.0

# Overnight window for "did this print finish overnight" and "did this
# email arrive overnight" framing. 14 hours covers a typical sleep window
# plus a buffer for late-night activity.
OVERNIGHT_WINDOW_SECONDS = 14 * 3600

# TTS budget. Spec is a 60-second briefing. Char-rate heuristic matches the
# v1 skill: Azure / Edge voices at the JARVIS rate land near 15 chars/s.
TTS_BUDGET_SECONDS   = 60.0
TTS_CHARS_PER_SECOND = 15.0

# Order in which sections are dropped when the briefing runs long. Lower
# priority items get dropped first. News goes first (background context that
# the user can grab from any feed); calendar is last (their actual day). Teams
# sits above print/deliveries because a pending human ping outranks an
# overnight printer status update.
TTS_DROP_ORDER = ("news", "weather", "deliveries", "print", "teams", "calendar")

_speech_lock = threading.Lock()
_state_lock  = threading.Lock()

# Tracks the rising edge of sustained presence within the watcher thread.
_presence_first_seen_at: list[float] = [0.0]


# ─── small helpers ───────────────────────────────────────────────────────

def _import_skill(name: str):
    """Best-effort import of a sibling skill — relative first, absolute fallback.

    Resolve the LIVE skill first: load_skills() registers each skill in
    sys.modules as ``skill_<name>``, and that copy holds the running
    poller's populated _state (e.g. face_tracker, bambu_monitor). A bare
    ``import_module(name)`` would load a SECOND, fresh copy whose _state is
    empty, so presence/print info would silently read nothing. Match
    daily_briefing / evening_briefing and prefer the already-registered
    module."""
    live = sys.modules.get(f"skill_{name}")
    if live is not None:
        return live
    try:
        return importlib.import_module(f"skills.{name}")
    except Exception:
        pass
    try:
        skills_dir = os.path.dirname(os.path.abspath(__file__))
        if skills_dir not in sys.path:
            sys.path.insert(0, skills_dir)
        return importlib.import_module(name)
    except Exception:
        return None


def _enqueue_speech(message: str) -> None:
    """Route announcements through bobert_companion.proactive_announce when
    available so the queue write goes through the canonical drainer. Falls
    back to a direct atomic write so the briefing still reaches
    pending_speech.json during early-boot skill registration."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="arrival_v2")
            return
    except Exception:
        pass

    queue_path = os.path.join(_PROJECT_DIR, "pending_speech.json")
    with _speech_lock:
        data = []
        if os.path.exists(queue_path):
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(queue_path, data)
        except Exception as e:
            print(f"  [arrival_v2] speech-queue write failed ({e}); briefing: {message}")


def _load_state() -> dict:
    with _state_lock:
        if not os.path.exists(_STATE_FILE):
            return {}
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}


def _save_state(state: dict) -> None:
    with _state_lock:
        try:
            _atomic_write_json(_STATE_FILE, state)
        except Exception as e:
            print(f"  [arrival_v2] state write failed: {e}")


def _already_fired_today() -> bool:
    return _load_state().get("last_fired_date") == time.strftime("%Y-%m-%d")


def _chain_morning_briefing_fired_today() -> bool:
    """True if any of morning_chain's single-fire morning briefings
    (arrival / handoff / briefing) already ran today.

    v2 runs its OWN presence watcher and is NOT one of morning_chain's
    SKILL_NAMES, so on a normal morning the user would get TWO briefings:
    whichever the chain picked, plus this one. To stay least-invasive we
    don't rewire the chain — we just consult the same on-disk same-day
    flags it tracks (reusing its _skill_already_fired_today so the date
    parsing for the JSON / raw-text state files stays in one place) and
    no-op when the day is already covered.

    Returns False on any failure — including when morning_chain isn't
    importable (the chain is disabled / not installed). That deliberately
    preserves the chain-disabled case: if nothing else fires today, none of
    those flags will be set, so this returns False and v2 still works."""
    try:
        mc = sys.modules.get("skill_morning_chain")
        if mc is None:
            mc = importlib.import_module("skills.morning_chain")
    except Exception:
        try:
            mc = importlib.import_module("morning_chain")
        except Exception:
            return False
    checker = getattr(mc, "_skill_already_fired_today", None)
    names = getattr(mc, "SKILL_NAMES", None)
    if not callable(checker) or not names:
        return False
    try:
        return any(checker(n) for n in names)
    except Exception:
        return False


def _mark_fired(reason: str) -> None:
    state = _load_state()
    state["last_fired_date"] = time.strftime("%Y-%m-%d")
    state["last_fired_ts"]   = time.time()
    state["last_reason"]     = reason
    _save_state(state)


# ─── presence detection ─────────────────────────────────────────────────

def _face_tracker_state() -> dict | None:
    """Snapshot face_tracker._state under its own lock. Returns None when
    face_tracker isn't importable or hasn't booted its poll thread yet —
    callers treat None as "presence unknown; do not fire"."""
    ft = _import_skill("face_tracker")
    if ft is None:
        return None
    snap_fn = getattr(ft, "_snapshot_state", None)
    if callable(snap_fn):
        try:
            snap = snap_fn()
            if isinstance(snap, dict):
                return snap
        except Exception:
            return None
    state = getattr(ft, "_state", None)
    lock  = getattr(ft, "_state_lock", None)
    if not isinstance(state, dict):
        return None
    if lock is not None:
        try:
            with lock:
                return dict(state)
        except Exception:
            return None
    return dict(state)


def _sustained_presence_seconds() -> float:
    """How long face has been continuously visible according to the
    rising-edge tracked by this watcher. 0.0 when not currently visible."""
    snap = _face_tracker_state()
    if snap is None:
        return 0.0
    if not snap.get("face_visible"):
        _presence_first_seen_at[0] = 0.0
        return 0.0
    last_sample = snap.get("last_sample_at") or 0.0
    # face_tracker hasn't polled yet → no usable signal.
    if not last_sample:
        return 0.0
    # Detect rising edge: first time we observe face_visible after a gap.
    if _presence_first_seen_at[0] == 0.0:
        # Anchor to last_face_at (the actual first sighting) when available,
        # not "now" — this lets the tracker credit time the face_tracker
        # already accumulated before this watcher started polling.
        first_face = snap.get("first_face_at") or 0.0
        last_face  = snap.get("last_face_at") or 0.0
        anchor = last_face if last_face else first_face
        _presence_first_seen_at[0] = anchor or time.time()
    return max(0.0, time.time() - _presence_first_seen_at[0])


def _within_morning_window() -> bool:
    hour = time.localtime().tm_hour
    return ARRIVAL_START_HOUR <= hour < ARRIVAL_END_HOUR


# ─── data-source sections ───────────────────────────────────────────────

def _section_weather() -> str:
    """Forward-looking weather alert for today. '' when no rain expected
    or weather chain unreachable."""
    wb = _import_skill("weather_briefing")
    if wb is None:
        return ""
    fn = getattr(wb, "get_umbrella_alert", None)
    if not callable(fn):
        return ""
    try:
        return (fn("today") or "").strip()
    except Exception as e:
        print(f"  [arrival_v2] weather fetch failed: {e}")
        return ""


def _section_teams() -> str:
    """Unread Microsoft Teams chat count with the top sender's first name.
    '' when nothing unread or ms_graph can't reach the Chat.Read endpoint."""
    ms = _import_skill("ms_graph")
    if ms is None:
        return ""
    fn = getattr(ms, "get_teams_unread_count", None)
    if not callable(fn):
        return ""
    try:
        result = fn()
    except Exception as e:
        print(f"  [arrival_v2] teams fetch failed: {e}")
        return ""
    if not isinstance(result, dict):
        return ""
    try:
        count = int(result.get("count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return ""
    top = (result.get("top_sender") or "").strip()
    if count == 1:
        return f"one new Teams message" + (f" from {top}" if top else "")
    if top:
        return f"{count} new Teams messages, one from {top}"
    return f"{count} new Teams messages"


def _section_news() -> str:
    """Top headlines from news_briefing (default 3). The intro the briefing
    skill normally prepends is stripped so this drops cleanly mid-monologue
    after 'Good morning, sir.' rather than re-greeting."""
    nb = _import_skill("news_briefing")
    if nb is None:
        return ""
    fn = getattr(nb, "get_news_text", None)
    if not callable(fn):
        return ""
    try:
        text = (fn() or "").strip()
    except Exception as e:
        print(f"  [arrival_v2] news fetch failed: {e}")
        return ""
    if not text:
        return ""
    text = re.sub(r"^Today's headlines,?\s*sir\.\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _section_print() -> str:
    """Overnight print completion or current print status. '' when the
    printer state isn't usable (no fresh MQTT samples, state too old)."""
    bm = _import_skill("bambu_monitor")
    summary_fn = getattr(bm, "get_last_print_completion_summary", None) if bm else None
    if callable(summary_fn):
        try:
            summary = summary_fn(within_seconds=OVERNIGHT_WINDOW_SECONDS)
        except Exception as e:
            print(f"  [arrival_v2] bambu summary failed: {e}")
            summary = None
        if isinstance(summary, dict):
            fname = (summary.get("filename") or "").strip()
            when  = (summary.get("finish_phrase") or "").strip()
            who   = f" of '{fname}'" if fname else ""
            if when:
                return f"the H2D finished{who} at {when} overnight"
            return f"the H2D finished{who} overnight"

    # Fall back to live state when no overnight finish was announced.
    state: dict = {}
    if bm is not None:
        lock = getattr(bm, "_state_lock", None)
        raw  = getattr(bm, "_state", None)
        if lock is not None and isinstance(raw, dict):
            try:
                with lock:
                    state = dict(raw)
            except Exception:
                state = {}
    if not state.get("last_update") and os.path.exists(_BAMBU_STATE):
        try:
            with open(_BAMBU_STATE, "r", encoding="utf-8") as f:
                state = json.load(f) or {}
        except Exception:
            state = {}

    last_update = state.get("last_update") or 0.0
    if not last_update:
        return ""
    age = time.time() - last_update
    if age > 36 * 3600:
        return ""

    gcode_state = (state.get("gcode_state") or "").upper()
    raw_name    = state.get("filename") or ""
    fname = ""
    if bm is not None and hasattr(bm, "_strip_filename"):
        try:
            fname = bm._strip_filename(raw_name)
        except Exception:
            fname = ""
    if not fname and raw_name:
        try:
            fname = re.sub(r"\.(3mf|gcode|gco)$", "",
                           os.path.basename(raw_name), flags=re.IGNORECASE)
            fname = re.sub(r"[_\-]+", " ", fname).strip()
        except Exception:
            fname = ""

    if gcode_state == "FAILED":
        return "the H2D's overnight print failed"
    if gcode_state in ("RUNNING", "PRINTING", "PREPARE"):
        pct = state.get("mc_percent")
        try:
            pct_int = int(pct)
        except (TypeError, ValueError):
            pct_int = None
        if pct_int is not None:
            return f"the H2D is mid-print at {pct_int} percent"
        return "the H2D is mid-print"
    if gcode_state == "PAUSE":
        return "the H2D is paused mid-print"
    return ""


def _section_deliveries() -> str:
    """Amazon orders / deliveries summary. '' when nothing is in transit."""
    aot = _import_skill("amazon_order_tracker")
    if aot is None:
        return ""
    fn = (getattr(aot, "action_check_orders", None)
          or getattr(aot, "check_orders", None))
    if not callable(fn):
        return ""
    try:
        result = (fn("") or "").strip()
    except Exception as e:
        print(f"  [arrival_v2] deliveries fetch failed: {e}")
        return ""
    if not result:
        return ""
    low = result.lower()
    # Sentinel responses from action_check_orders — silently drop.
    if (low.startswith("no active amazon")
            or low.startswith("nothing currently")):
        return ""
    return result


def _section_calendar() -> str:
    """Today's first meeting phrase. The planner specified schedule_manager
    as the source, but schedule_manager owns JARVIS's own scheduler — it
    does not expose calendar data. ms_graph.get_first_meeting() is JARVIS's
    canonical calendar source (and what v1 uses), so v2 reads from there
    for parity. Returns '' when calendar has no events today."""
    ms = _import_skill("ms_graph")
    if ms is None:
        return ""
    fn = getattr(ms, "get_first_meeting", None)
    if not callable(fn):
        return ""
    try:
        meeting = fn("today")
    except Exception as e:
        print(f"  [arrival_v2] calendar fetch failed: {e}")
        return ""
    if not meeting:
        return ""
    sdt = meeting.get("start")
    if not hasattr(sdt, "hour"):
        return ""

    disp_hour = sdt.hour % 12 or 12
    minute_part = f":{sdt.minute:02d}" if sdt.minute else ""
    suffix = "AM" if sdt.hour < 12 else "PM"

    subject   = (meeting.get("subject")   or "").strip()
    organizer = (meeting.get("organizer") or "").strip()

    who_label = ""
    if organizer and organizer.lower() not in (os.getenv("JARVIS_USER_NAME", "").lower(), "me"):
        first = organizer.split("<")[0].strip().split()
        if first and "@" not in first[0]:
            who_label = first[0]
    if subject and len(subject) <= 40:
        what = subject
    elif who_label:
        what = f"a sync with {who_label}"
    elif subject:
        what = subject
    else:
        what = "a meeting"

    if not minute_part and disp_hour >= 8 and sdt.hour < 12:
        return f"{what} at {disp_hour}"
    return f"{what} at {disp_hour}{minute_part} {suffix}"


# ─── parallel orchestration ─────────────────────────────────────────────

def _gather_sections() -> dict:
    """Run all six sections in parallel under SECTION_TIMEOUT_SECONDS each.
    A timeout or exception in any one section maps that section to '' so the
    briefing degrades gracefully rather than failing wholesale."""
    workers = {
        "weather":    _section_weather,
        "teams":      _section_teams,
        "print":      _section_print,
        "deliveries": _section_deliveries,
        "calendar":   _section_calendar,
        "news":       _section_news,
    }
    results: dict = {k: "" for k in workers}
    # NOT a `with` block: ThreadPoolExecutor.__exit__ calls shutdown(wait=True),
    # which would block the manual action on the SLOWEST section even though each
    # fut.result() already returned/timed out under SECTION_TIMEOUT_SECONDS. We
    # collect, then shutdown(wait=False) so the 8 s per-section timeout bounds the
    # wall clock. Unfinished sections are omitted (mapped to ''), as before.
    ex = ThreadPoolExecutor(max_workers=len(workers))
    try:
        futures = {ex.submit(fn): name for name, fn in workers.items()}
        for fut in list(futures):
            name = futures[fut]
            try:
                results[name] = (fut.result(timeout=SECTION_TIMEOUT_SECONDS) or "").strip()
            except FutureTimeoutError:
                print(f"  [arrival_v2] section '{name}' timed out — dropping")
                results[name] = ""
            except Exception as e:
                print(f"  [arrival_v2] section '{name}' crashed: {e}")
                results[name] = ""
    finally:
        # Don't join: cancel queued futures and let any still-running section
        # finish on its daemon worker without holding up the briefing.
        ex.shutdown(wait=False, cancel_futures=True)
    return results


def _compose_briefing(parts: dict) -> str:
    """Build the spoken briefing in JARVIS cadence. Each piece is independent
    — missing pieces collapse. Always leads with 'Good morning, sir.' since
    the trigger guarantees we're inside the morning window. Order matches the
    example monologue: ambient conditions → human comms → schedule → physical
    world → background news."""
    weather    = parts.get("weather", "")
    teams      = parts.get("teams", "")
    calendar   = parts.get("calendar", "")
    print_p    = parts.get("print", "")
    deliveries = parts.get("deliveries", "")
    news       = parts.get("news", "")

    sentences = ["Good morning, sir."]

    if weather:
        sentences.append(weather if weather.endswith(".") else weather + ".")
    if teams:
        line = teams[0].upper() + teams[1:]
        sentences.append(line if line.endswith(".") else line + ".")
    if calendar:
        sentences.append(f"You have {calendar}.")
    if print_p:
        sentences.append(print_p[0].upper() + print_p[1:] + ".")
    if deliveries:
        sentences.append(deliveries if deliveries.endswith(".") else deliveries + ".")
    if news:
        sentences.append(news if news.endswith(".") else news + ".")

    if len(sentences) == 1:
        # Nothing to brief on; surface that explicitly so the manual trigger
        # gives the caller a useful answer instead of an empty TTS payload.
        sentences.append("Nothing of note overnight.")

    return "[intent:briefing] " + " ".join(sentences)


def _estimate_tts_seconds(text: str) -> float:
    """Rough spoken-duration estimate for the 60-second budget. Strips the
    intent tag (not spoken) before counting."""
    body = re.sub(r"^\[intent:[^\]]+\]\s*", "", text or "")
    if not body:
        return 0.0
    return len(body) / TTS_CHARS_PER_SECOND


def _compose_within_budget(parts: dict) -> str:
    """Compose and progressively drop sections in TTS_DROP_ORDER until the
    estimated duration fits inside TTS_BUDGET_SECONDS."""
    text = _compose_briefing(parts)
    if _estimate_tts_seconds(text) <= TTS_BUDGET_SECONDS:
        return text
    trimmed = dict(parts)
    for drop_key in TTS_DROP_ORDER:
        if not trimmed.get(drop_key):
            continue
        trimmed[drop_key] = ""
        text = _compose_briefing(trimmed)
        if _estimate_tts_seconds(text) <= TTS_BUDGET_SECONDS:
            break
    return text


def _build_briefing() -> str:
    parts = _gather_sections()
    return _compose_within_budget(parts)


# ─── fire path ──────────────────────────────────────────────────────────

def _fire_arrival(reason: str, *, force: bool = False) -> str:
    """Build and enqueue the briefing. When force=False (auto path) bails if
    the briefing has already fired today — guards against TOCTOU between the
    watcher's pre-check, a manual invocation, and a chain dispatch."""
    if not force and _already_fired_today():
        print(f"  [arrival_v2] suppressing ({reason}) — already fired today")
        return ""
    # Don't double-brief: if morning_chain already fired one of its single-
    # fire morning briefings (arrival / handoff / briefing) today, stand down.
    # force=True (manual trigger) still bypasses. The chain-disabled case is
    # preserved — see _chain_morning_briefing_fired_today.
    if not force and _chain_morning_briefing_fired_today():
        print(f"  [arrival_v2] suppressing ({reason}) — morning_chain already "
              f"briefed today")
        return ""
    try:
        text = _build_briefing()
    except Exception as e:
        print(f"  [arrival_v2] briefing build failed: {e}")
        return ""
    if not text:
        return ""
    print(f"  [arrival_v2] queuing ({reason}): {text[:120]}...")
    _enqueue_speech(text)
    _mark_fired(reason)
    return text


def _fire_from_chain(reason: str = "morning_chain_v2") -> str:
    """Entry called by skills/morning_chain.py once it knows about v2. The
    chain's existing same-day-suppression and TOCTOU pattern applies — we
    just bail re-entrantly if the day's already covered."""
    if _already_fired_today():
        return ""
    return _fire_arrival(reason)


# ─── presence watcher ──────────────────────────────────────────────────

def _watch_for_arrival() -> None:
    """Daemon: poll face_tracker and fire on sustained presence inside the
    morning window. Same-day suppression keeps the briefing to one fire per
    calendar day even if the user walks in and out multiple times."""
    # Give face_tracker a head start so its first poll has landed.
    time.sleep(8.0)
    print(f"  [arrival_v2] presence watcher active "
          f"(window {ARRIVAL_START_HOUR}–{ARRIVAL_END_HOUR}, "
          f"sustained ≥ {SUSTAINED_PRESENCE_SECONDS:.1f}s)")

    while True:
        try:
            if _already_fired_today() or not _within_morning_window():
                time.sleep(WATCH_POLL_SECONDS)
                continue
            sustained = _sustained_presence_seconds()
            if sustained >= SUSTAINED_PRESENCE_SECONDS:
                _fire_arrival(
                    f"presence watcher (sustained {sustained:.1f}s)",
                )
        except Exception as e:
            print(f"  [arrival_v2] watcher tick error: {e}")
        time.sleep(WATCH_POLL_SECONDS)


# ─── registration ──────────────────────────────────────────────────────

def register(actions):
    def morning_arrival_v2(_: str = "") -> str:
        try:
            text = _fire_arrival("manual trigger", force=True)
            return text or "morning arrival v2 briefing produced no content"
        except Exception as e:
            return f"morning arrival v2 failed: {e}"

    actions["morning_arrival_v2"]   = morning_arrival_v2
    actions["arrival_briefing_v2"]  = morning_arrival_v2

    t = threading.Thread(
        target=_watch_for_arrival,
        daemon=True,
        name="morning-arrival-v2-watcher",
    )
    t.start()
