"""
Morning arrival skill for JARVIS — MCU-style cold-open briefing.

Fires on the first wake-word detection after 6+ hours of silence with a
single cohesive opener structured around "Three things require your
attention" — Teams DMs, Bambu print status, and overnight GPU/disk
anomalies — wrapped by time/weather and a calendar reminder when relevant.
The whole payload is budgeted under 25 seconds of TTS by dropping low-
priority sections (headline first, then claude merges, then meeting) if the
estimated duration exceeds the cap.

Actions added:
  morning_arrival     — manually trigger the cold-open briefing. Returns the
                        text (also queued for spoken delivery).

Auto-trigger:
  Driven by skills/morning_chain.py — a single controller polls
  bobert_companion._last_wake_date for the day's first wake event and picks
  ONE of {morning_arrival, morning_handoff, morning_briefing} to dispatch
  based on day-of-week / DEFAULT_MORNING_SKILL / time-of-day. When the chain
  picks arrival, it calls _fire_from_chain() here. The skill then verifies
  that bobert_companion.last_speech_time is at least MIN_SILENCE_HOURS old
  — this is the "morning arrival" gate; sub-6h gaps fall through silently so
  the chain can fall back to morning_briefing / morning_handoff on its next
  selection cycle. Same-day suppression is persisted via
  morning_arrival_state.json so a JARVIS restart doesn't re-fire.

Style: dry, JARVIS cadence, no preamble. Sources that fail silently degrade
to nothing — we'd rather skip a phrase than say 'I couldn't fetch X'.
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

from core.atomic_io import _atomic_write_json

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "morning_arrival_state.json")
_UPGRADE_LOG  = os.path.join(_PROJECT_DIR, "upgrade_stream.log")
_BAMBU_STATE  = os.path.join(_PROJECT_DIR, "bambu_overlay_state.json")
_SELF_DIAG_HISTORY = os.path.join(_PROJECT_DIR, "data", "self_diagnostic.json")

# "After sunrise" window — first wake whose local hour lands in [start, end)
# triggers the arrival opener. 6 AM matches morning_briefing's convention;
# the upper bound (12) keeps an after-lunch first wake from getting an out-
# of-place "Good morning" cold-open.
ARRIVAL_START_HOUR = 6
ARRIVAL_END_HOUR   = 12

# Minimum stretch of silence (no user speech) before the chain-dispatched
# arrival is allowed to fire. The spec describes the arrival opener as the
# "morning" greeting that fires on the first wake-word after a real overnight
# gap, not on every wake event inside [6, 12). Six hours conservatively
# covers a typical sleep window while still firing on a short late night.
MIN_SILENCE_HOURS = 6.0

# Delay after the wake greeting before the briefing queues, so JARVIS isn't
# speaking over its own "Good morning, sir." wake reply.
ARRIVAL_DELAY_SECONDS = 7.0

# Background poll interval for the wake-event watcher.
WATCH_POLL_SECONDS = 5.0

# Per-source hard timeout. Each section runs in parallel and any section that
# overruns is dropped from the brief — the cold open should not stall on a
# hung Graph or MQTT call.
SECTION_TIMEOUT_SECONDS = 8.0

# "Overnight" window for counting Claude Code merges, treating a finished
# Bambu print as "the H2D finished your bracket print", and scanning the
# self_diagnostic history for GPU/disk anomalies.
OVERNIGHT_WINDOW_SECONDS = 14 * 3600

# TTS-budget enforcement. The spec caps the cold-open at 25 s of spoken
# audio. We approximate spoken duration with a character-rate heuristic
# (Azure / Edge voices average ~150 chars per 10 s at the JARVIS rate) and
# drop sections in priority order when the estimate runs over.
TTS_BUDGET_SECONDS    = 25.0
TTS_CHARS_PER_SECOND  = 15.0
# Order in which sections are dropped when the briefing runs long. Items
# earlier in the list go first — the three "attention items" (Teams, Bambu,
# anomalies) are deliberately absent because they're the briefing's reason
# for being.
TTS_DROP_ORDER = ("headline", "claude", "meeting", "weather")

_speech_lock = threading.Lock()
_state_lock  = threading.Lock()


# ─── small helpers ───────────────────────────────────────────────────────

def _import_skill(name: str):
    """Best-effort import of a sibling skill — relative first, then absolute
    via the skills directory on sys.path.

    Resolve the LIVE skill first: load_skills() registers each skill in
    sys.modules as ``skill_<name>``, and that copy holds the running
    poller's populated _state. A bare ``import_module(name)`` would load a
    SECOND, fresh copy whose _state is empty, so this briefing would read
    nothing. Match daily_briefing / evening_briefing and prefer the
    already-registered module."""
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
    """Funnel atomic writes through bobert_companion.proactive_announce when
    available, falling back to a direct atomic write so the cold-open still
    reaches pending_speech.json during early-boot skill registration before
    the parent module has finished importing."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="arrival")
            return
    except Exception:
        pass

    with _speech_lock:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            print(f"  [arrival] speech-queue write failed ({e}); briefing: {message}")


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
            print(f"  [arrival] state write failed: {e}")


def _arrival_already_fired_today() -> bool:
    state = _load_state()
    return state.get("last_fired_date") == time.strftime("%Y-%m-%d")


def _silence_hours_since_last_speech() -> float | None:
    """Hours of silence at the moment the most recent wake event fired.

    Prefers bobert_companion._pre_wake_silence_seconds[0], which captures
    the gap from `last_speech_time` AT WAKE TIME — before the greeting TTS
    bumps the timer. Falling back to live `last_speech_time` would always
    read ~0 by the time the morning chain dispatches us (JARVIS just spoke
    "Good morning, sir." in response to the wake), defeating the silence
    gate entirely.

    Returns None when neither value is reachable. Callers should treat
    None as "can't decide — don't block" (i.e. degrade open / fire).
    """
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return None

    pre_wake = getattr(bc, "_pre_wake_silence_seconds", None)
    if isinstance(pre_wake, list) and pre_wake:
        try:
            val = float(pre_wake[0])
            if val > 0:
                return val / 3600.0
        except (TypeError, ValueError):
            pass

    lst = getattr(bc, "last_speech_time", None)
    if not isinstance(lst, (int, float)) or lst <= 0:
        return None
    return max(0.0, (time.time() - float(lst)) / 3600.0)


def _mark_fired(reason: str) -> None:
    state = _load_state()
    state["last_fired_date"] = time.strftime("%Y-%m-%d")
    state["last_fired_ts"]   = time.time()
    state["last_reason"]     = reason
    _save_state(state)


# ─── data-source sections (all return short phrases or '') ───────────────

def _section_time_phrase() -> str:
    """'7:42 AM' style local time. Always non-empty — time never fails."""
    now = time.localtime()
    disp_hour = now.tm_hour % 12 or 12
    suffix = "AM" if now.tm_hour < 12 else "PM"
    return f"{disp_hour}:{now.tm_min:02d} {suffix}"


def _section_weather_phrase() -> str:
    """'64 degrees and clear' style. '' on total weather-chain failure.

    Reuses skills/briefing_sources.get_weather_data() so the wttr → Open-Meteo
    → cached-last-known fallback chain protects us when wttr is unreachable.
    """
    bs = _import_skill("briefing_sources")
    if not bs:
        return ""
    try:
        data = bs.get_weather_data()
    except Exception as e:
        print(f"  [arrival] weather fetch failed: {e}")
        return ""
    if not data:
        return ""
    try:
        temp_c = int(data["temp_c"])
    except (KeyError, TypeError, ValueError):
        return ""
    # Convert C → F for the spec's "64 degrees" phrasing (which is clearly
    # Fahrenheit — 64 C would be a survival emergency, not a morning brief).
    temp_f = int(round(temp_c * 9 / 5 + 32))
    desc = (data.get("desc") or "").strip().lower()
    if not desc:
        return f"{temp_f} degrees"
    return f"{temp_f} degrees and {desc}"


def _section_first_meeting_phrase() -> str:
    """'a Sam sync at 10' style. '' when calendar has no events today."""
    ms = _import_skill("ms_graph")
    if not ms:
        return ""
    try:
        meeting = ms.get_first_meeting("today")
    except Exception as e:
        print(f"  [arrival] graph meeting fetch failed: {e}")
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
    # Subject often carries the more meaningful label ("Sam sync") than
    # the organizer ("Alex Morgan"). Prefer subject when it's short.
    if subject and len(subject) <= 40:
        what = subject
    elif who_label:
        what = f"a sync with {who_label}"
    elif subject:
        what = subject
    else:
        what = "a meeting"

    # For on-the-hour meetings, drop the ":00" — "a Sam sync at 10" reads
    # better than "at 10:00 AM". Keep the suffix only when ambiguous (PM).
    if not minute_part and disp_hour >= 8 and sdt.hour < 12:
        return f"{what} at {disp_hour}"
    return f"{what} at {disp_hour}{minute_part} {suffix}"


def _count_overnight_merges() -> int:
    """Count upgrade-loop iterations whose timestamp falls inside the
    overnight window. Best-effort: the upgrade_stream.log format is fragile
    (parsed via the `=== upgrade loop started YYYY-MM-DD HH:MM:SS ===` and
    `[loop] iteration N of M — K task(s) remain` markers), so we return 0
    rather than crashing the whole briefing if the format shifts."""
    if not os.path.exists(_UPGRADE_LOG):
        return 0
    cutoff = time.time() - OVERNIGHT_WINDOW_SECONDS
    try:
        size = os.path.getsize(_UPGRADE_LOG)
        # Read at most the tail of the log — overnight runs rarely produce
        # more than a few hundred KB and we don't need ancient history.
        read_bytes = min(size, 512 * 1024)
        with open(_UPGRADE_LOG, "rb") as f:
            if size > read_bytes:
                f.seek(size - read_bytes)
                # Skip a partial leading line
                f.readline()
            tail = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [arrival] upgrade log read failed: {e}")
        return 0

    # Walk the tail forward and count iterations whose enclosing
    # `=== upgrade loop started TS ===` is in-window. We treat each
    # `[loop] iteration N of M` line as one task-merge attempt; final
    # completions are accompanied by a decrement of "K task(s) remain", so
    # the count of distinct iteration lines is a good proxy for "improvements
    # merged overnight" without requiring git access.
    ts_re   = re.compile(r"^=== upgrade loop started (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===")
    iter_re = re.compile(r"^\[loop\] iteration \d+ of \d+ — (\d+) task\(s\) remain")
    current_in_window = False
    prev_remain: int | None = None
    merges = 0
    for line in tail.splitlines():
        m = ts_re.match(line)
        if m:
            try:
                t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                current_in_window = t >= cutoff
            except Exception:
                current_in_window = False
            prev_remain = None
            continue
        if not current_in_window:
            continue
        m = iter_re.match(line)
        if not m:
            continue
        try:
            remain = int(m.group(1))
        except ValueError:
            continue
        # Increment on a strict decrement of the "K task(s) remain" counter —
        # this matches what the user would observe as "Claude Code merged X
        # improvements overnight" rather than just "X iterations ran".
        if prev_remain is not None and remain < prev_remain:
            merges += (prev_remain - remain)
        prev_remain = remain
    return merges


def _section_claude_code_phrase() -> str:
    """'Claude Code merged 3 improvements' style; '' if there were none."""
    n = _count_overnight_merges()
    if n <= 0:
        return ""
    if n == 1:
        return "Claude Code merged one improvement"
    return f"Claude Code merged {n} improvements"


def _section_print_phrase() -> str:
    """'the H2D finished your bracket print' / 'the H2D is mid-print at 47
    percent' style; '' when the printer state isn't usable.

    Prefers bambu_monitor's in-process state when the poller is live, but
    falls back to reading bambu_overlay_state.json on disk so the cold-open
    still has Bambu data when JARVIS just started and the poller hasn't
    populated _state yet."""
    state: dict = {}
    bm = _import_skill("bambu_monitor")
    if bm is not None:
        try:
            with bm._state_lock:
                state = dict(bm._state)
        except Exception:
            state = {}
    # Fallback to the on-disk overlay state when in-process state is empty
    # (e.g. fresh JARVIS launch before the MQTT poller's first report).
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
    # Reading a >36h-old state is misleading — likely the printer's been off.
    if age > 36 * 3600:
        return ""

    gcode_state = (state.get("gcode_state") or "").upper()
    raw_name    = state.get("filename") or ""
    fname = ""
    try:
        if bm is not None:
            fname = bm._strip_filename(raw_name)
        else:
            fname = re.sub(r"\.(3mf|gcode|gco)$", "", os.path.basename(raw_name), flags=re.IGNORECASE)
            fname = re.sub(r"[_\-]+", " ", fname).strip()
    except Exception:
        fname = ""

    if gcode_state == "FINISH" and age <= OVERNIGHT_WINDOW_SECONDS:
        if fname:
            return f"the H2D finished your {fname} print"
        return "the H2D finished its overnight print"
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


def _section_teams_phrase() -> str:
    """Teams notification phrase. Tries a vision pass that asks Claude to
    extract the top 3 sender names from the Teams sidebar — this is more
    informative than just an unread count for the arrival cold-open. Falls
    back to teams_nudge's plain unread-count detector when the top-3 pass
    times out, can't parse, or returns NONE.

    Result examples (in priority order):
      'three new Teams chats from Alex, Sam, and Pat'
      'two new Teams chats from Alex and Sam'
      'one new Teams chat from Alex'
      '3 unread Teams messages including one from Alex'   (fallback)
      '2 unread Teams messages'                            (fallback)
      ''                                                   (nothing unread)
    """
    senders = _teams_top_senders_via_vision()
    if senders:
        n = len(senders)
        if n == 1:
            return f"one new Teams chat from {senders[0]}"
        if n == 2:
            return f"two new Teams chats from {senders[0]} and {senders[1]}"
        return (f"three new Teams chats from "
                f"{senders[0]}, {senders[1]}, and {senders[2]}")

    # Fallback to teams_nudge's existing unread-count detector.
    tn = _import_skill("teams_nudge")
    if not tn:
        return ""
    try:
        has_unread, count, sender = tn._ask_vision_for_teams_state()
    except Exception as e:
        print(f"  [arrival] teams vision lookup failed: {e}")
        return ""
    if not has_unread or count <= 0:
        return ""
    sender = (sender or "").strip()
    if count == 1:
        if sender:
            return f"one unread Teams message from {sender}"
        return "one unread Teams message"
    if sender:
        return f"{count} unread Teams messages including one from {sender}"
    return f"{count} unread Teams messages"


def _teams_top_senders_via_vision() -> list[str]:
    """Ask Claude to read the top 3 unread Teams sidebar chats and return
    the sender names. Returns [] on any failure — caller falls back to the
    plain unread-count detector.

    Re-uses bobert_companion.take_all_monitor_screenshots +
    ask_vision_multi (same plumbing teams_nudge uses) instead of bringing
    in new dependencies. The prompt is deliberately strict about output
    format so a hand-wavy vision reply doesn't corrupt the briefing.
    """
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return []
    take = getattr(bc, "take_all_monitor_screenshots", None)
    ask  = getattr(bc, "ask_vision_multi", None)
    if not callable(take) or not callable(ask):
        return []
    try:
        images = take()
    except Exception:
        return []
    if not images:
        return []
    prompt = (
        "Look across all monitors for the Microsoft Teams sidebar (Chat / "
        "Activity feed). Identify up to THREE unread chat or notification "
        "entries — rows with a bold sender name and an unread count badge, "
        "or notifications listed at the top of the Activity feed.\n\n"
        "Respond on ONE LINE in EXACTLY this format:\n"
        "  TOP: name1 | name2 | name3\n"
        "  TOP: name1\n"
        "  NONE\n"
        "Use first names only (drop last names) and trim to at most 14 "
        "characters per name. Reply 'NONE' if Teams is not visible or has "
        "no unread sidebar entries. Do NOT add any explanation."
    )
    try:
        answer = ask(prompt, images)
    except Exception:
        return []
    if not isinstance(answer, str):
        return []
    line = answer.strip().splitlines()[0] if answer.strip() else ""
    upper = line.upper()
    if not line or upper.startswith("NONE") or "TOP:" not in upper:
        return []
    payload = line.split(":", 1)[1].strip() if ":" in line else ""
    if not payload:
        return []
    names: list[str] = []
    for raw in payload.split("|"):
        n = re.sub(r"[^A-Za-z .'-]", "", raw).strip()
        # Drop placeholder values / overly long blobs that signal the
        # model didn't actually find sidebar entries.
        if not n or n.upper() in ("NONE", "N/A", "UNKNOWN"):
            continue
        if len(n) > 14:
            n = n.split()[0][:14]
        names.append(n)
        if len(names) >= 3:
            break
    return names


def _section_overnight_anomalies() -> str:
    """GPU / disk anomalies pulled from the self_diagnostic history.

    Reads the history file written by skills/self_diagnostic.py
    (data/self_diagnostic.json), scans runs inside the overnight window,
    and reports up to two anomalies. Returns '' when the diag didn't run
    overnight, the file is missing/corrupt, or no GPU/disk probe failed.

    Output examples:
      'a GPU anomaly overnight'
      'a disk anomaly overnight'
      'GPU and disk anomalies overnight'
    """
    if not os.path.exists(_SELF_DIAG_HISTORY):
        return ""
    try:
        with open(_SELF_DIAG_HISTORY, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        print(f"  [arrival] self_diag history read failed: {e}")
        return ""
    # The file is normally a list of run dicts but the loader in
    # self_diagnostic also accepts {"runs": [...]}. Handle both shapes.
    if isinstance(history, dict):
        history = history.get("runs") or []
    if not isinstance(history, list):
        return ""

    cutoff = time.time() - OVERNIGHT_WINDOW_SECONDS
    seen: dict[str, bool] = {}  # component → has-anomaly flag
    for run in history:
        if not isinstance(run, dict):
            continue
        try:
            ts = float(run.get("ts") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts < cutoff:
            continue
        probes = run.get("probes") or {}
        for component in ("gpu", "disk"):
            if seen.get(component):
                continue
            probe = probes.get(component) or {}
            if probe.get("ok"):
                continue
            # Only flag as an anomaly if the probe actually failed AND the
            # severity isn't LOW (LOW = transient / environmental — not
            # worth waking the user about).
            sev = (probe.get("severity") or "").upper()
            if sev == "LOW":
                continue
            err = (probe.get("error") or "").strip()
            if not err:
                continue
            seen[component] = True

    flagged = [c for c in ("gpu", "disk") if seen.get(c)]
    if not flagged:
        return ""
    # "GPU" is an acronym (uppercase) but "disk" is a common noun (lower).
    label = {"gpu": "GPU", "disk": "disk"}
    if len(flagged) == 1:
        return f"a {label[flagged[0]]} anomaly overnight"
    return "GPU and disk anomalies overnight"


def _section_top_headline() -> str:
    """Single top headline sentence, no leading 'Today's headlines, sir.'
    intro. '' when news is disabled or feeds failed.

    We deliberately don't reuse news_briefing.get_news_text() because it
    returns the multi-headline paragraph with its own intro line — the
    arrival cold-open only wants ONE headline summarised in the closer.
    Instead we pull the first headline directly from news_briefing's
    internal _gather_headlines helper. If that helper isn't available we
    silently drop the news leg."""
    nb = _import_skill("news_briefing")
    if not nb:
        return ""
    try:
        cfg = nb._read_config()
        if not cfg.get("enabled") or not cfg.get("feeds"):
            return ""
        # Borrow the config but force count=1 for the arrival cold-open
        cfg = dict(cfg)
        cfg["count"] = 1
        headlines = nb._gather_headlines(cfg)
    except Exception as e:
        print(f"  [arrival] news fetch failed: {e}")
        return ""
    if not headlines:
        return ""

    h = headlines[0]
    try:
        if cfg.get("summarize"):
            line = nb._summarize_via_llm(h.get("title", ""), h.get("description", ""))
        else:
            line = h.get("title", "")
    except Exception:
        line = h.get("title", "")
    line = (line or "").strip().rstrip(".")
    return line


# ─── parallel orchestration + composition ───────────────────────────────

def _gather_sections() -> dict:
    """Run all data-source sections in parallel and collect their results
    inside SECTION_TIMEOUT_SECONDS each. Any section that times out or
    raises is replaced with '' so the cold-open never stalls on a slow leg."""
    workers = {
        "time":      _section_time_phrase,
        "weather":   _section_weather_phrase,
        "meeting":   _section_first_meeting_phrase,
        "claude":    _section_claude_code_phrase,
        "print":     _section_print_phrase,
        "teams":     _section_teams_phrase,
        "anomalies": _section_overnight_anomalies,
        "headline":  _section_top_headline,
    }
    results: dict = {k: "" for k in workers}
    # NOT a `with` block: ThreadPoolExecutor.__exit__ calls shutdown(wait=True),
    # which would block the manual action on the SLOWEST section (~30 s via
    # ask_vision_multi) even though each fut.result() already returned/timed out.
    # We collect under the per-section timeout, then shutdown(wait=False) so the
    # 8 s SECTION_TIMEOUT_SECONDS bounds the wall clock. Sections that didn't
    # finish in time are simply omitted (mapped to ''), identical to before.
    ex = ThreadPoolExecutor(max_workers=len(workers))
    try:
        futures = {ex.submit(fn): name for name, fn in workers.items()}
        for fut in list(futures):
            name = futures[fut]
            try:
                results[name] = (fut.result(timeout=SECTION_TIMEOUT_SECONDS) or "").strip()
            except FutureTimeoutError:
                print(f"  [arrival] section '{name}' timed out — dropping")
                results[name] = ""
            except Exception as e:
                print(f"  [arrival] section '{name}' crashed: {e}")
                results[name] = ""
    finally:
        # Don't join: cancel queued futures and let any still-running section
        # finish on its daemon worker without holding up the briefing.
        ex.shutdown(wait=False, cancel_futures=True)
    return results


def _attention_clause(items: list[str]) -> str:
    """Format Teams / Bambu / anomalies as the spec's 'Three things require
    your attention' clause. Item count flexes from 1–3."""
    n = len(items)
    if n == 0:
        return ""
    if n == 1:
        return f"One item requires your attention: {items[0]}."
    if n == 2:
        return f"Two things require your attention: {items[0]}, and {items[1]}."
    return (f"Three things require your attention: "
            f"{items[0]}, {items[1]}, and {items[2]}.")


def _compose_briefing(parts: dict) -> str:
    """Compose the cold-open in the spec's JARVIS cadence, grouping the
    three priority items (Teams, Bambu, overnight anomalies) under the
    'Three things require your attention' line as the briefing's spine.

    Shape (each piece independently optional — missing pieces collapse):
      1. 'Good morning, sir.'
      2. 'It's <time>, <weather>.'
      3. 'Three things require your attention: <teams>, <print>, and <anomalies>.'
      4. 'You have <meeting>.'
      5. 'Overnight, <claude>.'
      6. 'In the news, <headline>.'
    """
    time_p     = parts.get("time", "")
    weather    = parts.get("weather", "")
    meeting    = parts.get("meeting", "")
    claude     = parts.get("claude", "")
    print_p    = parts.get("print", "")
    teams      = parts.get("teams", "")
    anomalies  = parts.get("anomalies", "")
    headline   = parts.get("headline", "")

    sentences = ["Good morning, sir."]

    s2_bits = []
    if time_p:
        s2_bits.append(f"It's {time_p}")
    if weather:
        s2_bits.append(weather)
    if s2_bits:
        sentences.append(", ".join(s2_bits) + ".")

    attention_items = [s for s in (teams, print_p, anomalies) if s]
    clause = _attention_clause(attention_items)
    if clause:
        sentences.append(clause)

    if meeting:
        sentences.append(f"You have {meeting}.")
    if claude:
        sentences.append(f"Overnight, {claude}.")
    if headline:
        sentences.append(f"In the news, {headline}.")

    return "[intent:briefing] " + " ".join(sentences)


def _estimate_tts_seconds(text: str) -> float:
    """Rough estimate of spoken duration for the TTS budget gate. Strips
    the leading [intent:...] tag (not spoken) before counting characters
    at TTS_CHARS_PER_SECOND. Off by 10–20% in practice — fine for a soft
    cap intended to prevent a 60-second monologue."""
    body = re.sub(r"^\[intent:[^\]]+\]\s*", "", text or "")
    if not body:
        return 0.0
    return len(body) / TTS_CHARS_PER_SECOND


def _compose_within_budget(parts: dict) -> str:
    """Compose the briefing and progressively drop sections (in TTS_DROP_ORDER)
    until the estimated spoken duration fits inside TTS_BUDGET_SECONDS. The
    three attention items are never dropped — they're the briefing's spine."""
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


# ─── fire path ───────────────────────────────────────────────────────────

def _fire_arrival(reason: str, *, force: bool = False) -> str:
    """Build and enqueue the cold-open. When `force` is False (auto-trigger
    path), bail with '' if the briefing has already fired today — defends
    against a TOCTOU race between the register-time gate, the watcher's
    pre-check, and a parallel manual invocation. Manual triggers always run."""
    if not force and _arrival_already_fired_today():
        print(f"  [arrival] suppressing ({reason}) — already fired today")
        return ""
    try:
        text = _build_briefing()
    except Exception as e:
        print(f"  [arrival] briefing build failed: {e}")
        return ""
    print(f"  [arrival] queuing ({reason}): {text[:120]}...")
    _enqueue_speech(text)
    _mark_fired(reason)
    return text


# ─── chain entry point ───────────────────────────────────────────────────

def _fire_from_chain(reason: str = "morning_chain") -> str:
    """Auto-trigger entry called by skills/morning_chain.py once it has
    decided arrival is today's pick. Preserves the original watcher's
    TOCTOU-safe pattern verbatim: pre-check → delay → re-check → fire.

    Additionally enforces the spec's silence-based gate: we only fire the
    "Good morning" cold-open when bobert_companion.last_speech_time shows
    at least MIN_SILENCE_HOURS of silence. If the gate fails the skill
    returns '' silently so the morning chain can fall back to its next
    pick (handoff / briefing) for short late-night gaps. Manual triggers
    ("morning arrival") still bypass everything via force=True.
    """
    if _arrival_already_fired_today():
        return ""
    silence_h = _silence_hours_since_last_speech()
    # silence_h is None when bobert_companion isn't reachable — in that case
    # we degrade open (fire) rather than closed (skip) so a freshly-booted
    # JARVIS with no recorded speech still gets its arrival briefing.
    if silence_h is not None and silence_h < MIN_SILENCE_HOURS:
        print(f"  [arrival] silence gate failed — last speech "
              f"{silence_h:.1f}h ago < {MIN_SILENCE_HOURS:.1f}h; "
              f"deferring to chain's next pick")
        return ""
    time.sleep(ARRIVAL_DELAY_SECONDS)
    if _arrival_already_fired_today():
        return ""
    return _fire_arrival(reason)


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    def morning_arrival(_: str = "") -> str:
        try:
            text = _fire_arrival("manual trigger", force=True)
            return text or "morning arrival briefing produced no content"
        except Exception as e:
            return f"morning arrival failed: {e}"

    actions["morning_arrival"] = morning_arrival
    actions["arrival_briefing"] = morning_arrival  # natural alias

    # Auto-trigger is owned by skills/morning_chain.py — no per-skill watcher.
