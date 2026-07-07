"""
Proactive print companion — JARVIS-voice commentary on Bambu H2D prints.

Rides on top of skills/bambu_monitor.py:
  * Polls bambu_monitor._state every POLL_INTERVAL_SECONDS on a daemon
    thread (read-only — bambu_monitor owns the writes).
  * At 25/50/75% completion, emits a wry MCU-style observation with an
    ETA-derived completion time. Trails bambu_monitor's plain 25/50/75%
    call by MILESTONE_OFFSET_SECONDS so the two announcements don't
    stack — bambu_monitor speaks first, this skill follows up with the
    flavour line a beat later.
  * On print FINISH (after observing this print actually running),
    offers to dim the workshop lights and queue a 30-minute cooldown
    timer. Both are tentative — the user must accept before either
    fires; we never auto-invoke smart_home_control / set_timer.
  * Persists per-print metadata to
    data/proactive_print_companion_patterns.json, bucketed by
    (material, layer-count). On every new print, surfaces a warning
    when the historical failure rate for similar prints is high.
  * Optionally samples the local vision skill at each milestone to
    look for visible failure signatures (stringy extrusion, clog,
    adhesion loss, layer shift). Degrades silently when local vision
    isn't available — the rest of the skill stays useful offline.

Pattern file is bounded by PER_BUCKET_RETENTION (most-recent-N per
bucket) so it doesn't grow without limit as prints accumulate.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
import threading
import time

_PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR      = os.path.join(_PROJECT_DIR, "data")
_PATTERNS_FILE = os.path.join(_DATA_DIR, "proactive_print_companion_patterns.json")

# Ensure project root is importable so core.atomic_io resolves whether
# this module is loaded as `skill_proactive_print_companion` or directly.
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402


# ── tuning ──────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS    = 15.0
# Give bambu_monitor's MQTT client time to connect and populate _state
# before our first peek; a fresh _state would otherwise look like an
# idle printer and we'd skip the first poll's setup work.
INITIAL_DELAY_SECONDS    = 45.0
# After we first detect a milestone crossing, wait this many seconds
# before speaking our companion line so bambu_monitor's parallel
# "Print is 25% complete" line gets spoken first. ~45s ≥ three polls
# is comfortably longer than bambu_monitor's MQTT push cadence.
MILESTONE_OFFSET_SECONDS = 45.0
# Per-bucket retention — keep the most recent N records so the file
# stays bounded across hundreds of prints.
PER_BUCKET_RETENTION              = 50
MIN_HISTORY_FOR_WARNING           = 3
HISTORICAL_FAILURE_WARN_THRESHOLD = 0.30
# Coarse buckets keep "similar" prints aligned (a 215 and 250 layer
# print land together) without diluting the failure-rate signal.
LAYER_BUCKET_SIZE = 100
# At print start we treat anything past this percentage as a
# mid-flight discovery — duration_min would be wrong if we recorded
# it, so we suppress the history write for those prints.
MIDFLIGHT_PCT_THRESHOLD = 5.0

# Failure signatures we look for in the VLM's short reply. Anything
# matching here at milestone-time surfaces a proactive warning.
_VISION_FAILURE_KEYWORDS = (
    "spaghetti", "stringy", "string", "clog", "nozzle clog",
    "adhesion", "lifting", "warp", "shifted", "layer shift",
    "missing", "blob", "tangled", "detached",
)

# MCU-style hedging tails cycled through milestone commentary so
# consecutive prints don't sound identical.
_MCU_HEDGE_SUFFIXES = (
    "accounting for a potential hiccup in the cooling fans",
    "give or take a wobble in the filament tension",
    "barring any unexpected chamber swing",
    "assuming the nozzle behaves itself",
    "give or take a touch of cooling latency",
)


# ── module-level state ─────────────────────────────────────────────
_stop_evt = threading.Event()
_thread   = [None]

# _poll_once is invoked from BOTH our own poll loop AND bambu_monitor's
# state-change hook (which fires on paho-mqtt's network thread). Without
# serialisation the two callers can both pass an
# "_announced_*"/"milestone-not-set" check before either sets it, so a
# finish/milestone line double-fires. Hold this lock across the whole
# read-modify-write body — mirrors bambu_h2d_voice_companion's _state_lock.
_poll_lock = threading.Lock()

# Per-print bookkeeping. Reset on every new_print transition.
_current_filename            = [None]
_last_gcode_state            = [None]
_print_start_ts              = [0.0]
_print_started_pct           = [None]
_print_midflight             = [False]
_saw_running_this_print      = [False]
_milestone_detected_at: dict = {}
_milestone_announced: set    = set()
_announced_completion_offer  = [False]
_warned_historical_failure   = [False]
_vision_samples: list        = []
_mcu_hedge_idx               = [0]


# ── bambu_monitor bridge ───────────────────────────────────────────
def _get_bambu_module():
    """Lazily resolve skill_bambu_monitor from sys.modules so we never
    force an import order. Returns None when bambu isn't loaded."""
    return sys.modules.get("skill_bambu_monitor")


def _read_state():
    """Snapshot bambu_monitor._state under its lock. Returns None when
    bambu isn't loaded so callers can degrade cleanly."""
    bm = _get_bambu_module()
    if bm is None:
        return None
    try:
        with bm._state_lock:
            return dict(bm._state)
    except Exception:
        return None


def _bambu_already_announced(threshold: int) -> bool:
    """True when bambu_monitor has already fired its own 25/50/75% line
    for this threshold — used to gate our companion line so we trail
    rather than stack."""
    bm = _get_bambu_module()
    if bm is None:
        return False
    try:
        announced = getattr(bm, "_announced_milestones", None)
        if isinstance(announced, set):
            return threshold in announced
    except Exception:
        pass
    return False


def _strip_filename(name: str) -> str:
    """Prefer bambu_monitor's filename stripper so we render identically
    to the rest of the H2D voice copy; fall back to a local copy when
    bambu hasn't loaded yet."""
    bm = _get_bambu_module()
    if bm is not None and hasattr(bm, "_strip_filename"):
        try:
            return bm._strip_filename(name)
        except Exception:
            pass
    if not name:
        return ""
    base = os.path.basename(name)
    base = re.sub(r"\.(3mf|gcode|gco)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_\-]+", " ", base).strip()
    return base[:60] if base else ""


# ── speech ────────────────────────────────────────────────────────
def _enqueue_speech(message: str) -> None:
    """Route a proactive line through bobert_companion's
    proactive_announce(), falling back to bambu_monitor's queue writer
    and ultimately a direct write so we never silently lose a callout."""
    try:
        bc = importlib.import_module("bobert_companion")
        fn = getattr(bc, "proactive_announce", None)
        if callable(fn):
            fn(message, source="print_companion")
            return
    except Exception:
        pass
    bm = _get_bambu_module()
    if bm is not None and hasattr(bm, "_enqueue_speech"):
        try:
            bm._enqueue_speech(message)
            return
        except Exception:
            pass
    queue_path = os.path.join(_PROJECT_DIR, "pending_speech.json")
    try:
        data = []
        if os.path.exists(queue_path):
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        if not isinstance(data, list):
            data = []
        data.append({"ts": time.time(), "message": message})
        _atomic_write_json(queue_path, data)
    except Exception as e:
        print(f"  [print_companion] speech-queue write failed ({e}); "
              f"alert: {message}")


# ── helpers ───────────────────────────────────────────────────────
def _format_eta_clock(remaining_min) -> str:
    """Render an HH:MM completion time from minutes-remaining. Empty
    string when the estimate isn't available so callers can fall back
    to a vaguer phrasing."""
    try:
        m = int(remaining_min)
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return ""
    target = time.localtime(time.time() + m * 60)
    return f"{target.tm_hour:02d}:{target.tm_min:02d}"


def _infer_material(filename: str) -> str:
    """Best-effort material extraction from a Bambu gcode filename.
    Bambu Studio commonly bakes the filament type into the name
    (e.g. `Bracket_PLA_BLACK_4h30m.3mf`); fall back to 'unknown' when
    nothing recognisable is in the name."""
    if not filename:
        return "unknown"
    name = filename.lower()
    # Order matters — "pla+" must match before "pla" so the plus
    # variant doesn't collapse into the base material bucket.
    candidates = ("pla+", "petg", "abs+", "asa", "tpu", "nylon",
                  "carbon", "wood", "silk", "pla", "abs")
    for tag in candidates:
        pat = rf"(?:^|[_\- ]){re.escape(tag)}(?:[_\- ]|\.|$)"
        if re.search(pat, name):
            return tag.replace("+", "plus")
    return "unknown"


def _bucket_key(material: str, total_layer) -> str:
    try:
        layers = int(total_layer)
    except (TypeError, ValueError):
        layers = 0
    bucket = (layers // LAYER_BUCKET_SIZE) * LAYER_BUCKET_SIZE
    return f"{material}_{bucket}"


def _next_hedge_suffix() -> str:
    idx = _mcu_hedge_idx[0] % len(_MCU_HEDGE_SUFFIXES)
    _mcu_hedge_idx[0] = idx + 1
    return _MCU_HEDGE_SUFFIXES[idx]


def _milestone_commentary(threshold: int, remaining_min) -> str:
    """Build the wry MCU-style milestone line."""
    eta = _format_eta_clock(remaining_min)
    eta_phrase = (f"current trajectory suggests completion at {eta}"
                  if eta else "trajectory still settling")
    hedge = _next_hedge_suffix()
    if threshold == 25:
        opener = "Quarter of the way along, sir"
    elif threshold == 50:
        opener = "Halfway through, sir"
    else:
        opener = "Three-quarters complete, sir"
    return f"{opener} — {eta_phrase}, {hedge}."


def _light_skill_available() -> bool:
    """True if sh_hue or sh_govee is loaded AND reports at least one device.

    Guards the "Shall I dim the workshop lights" offer so we never volunteer
    to dim what isn't there. is_available() probes alone aren't enough —
    sh_hue.is_available() returns True before the bridge has been paired,
    so we additionally require list_devices() to surface at least one bulb.
    Errors during the probe degrade to "unavailable" rather than crash.
    """
    for modname in ("skill_sh_hue", "skill_sh_govee"):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        try:
            is_avail = getattr(mod, "is_available", None)
            if callable(is_avail) and not is_avail():
                continue
        except Exception:
            continue
        try:
            list_fn = getattr(mod, "list_devices", None)
            if callable(list_fn):
                devs = list_fn()
                if devs:
                    return True
        except Exception:
            continue
    return False


def _timer_skill_available() -> bool:
    """True when skills/timer.py is loaded — checked by module presence and
    by the public set_timer action being wired into bobert_companion's
    ACTIONS dict (so we don't dangle a "queue a timer" offer the user can't
    accept)."""
    if sys.modules.get("skill_timer") is None:
        return False
    bc = sys.modules.get("bobert_companion") or sys.modules.get("__main__")
    actions = getattr(bc, "ACTIONS", None) if bc is not None else None
    if isinstance(actions, dict) and callable(actions.get("set_timer")):
        return True
    # Fall back to module-level presence — even if ACTIONS hasn't been
    # exposed (test harness, partial load), the timer module itself is
    # enough to dispatch through.
    return True


def _completion_offer_line(pretty_filename: str) -> str:
    """Compose the FINISH offer based on which downstream skills are live.

    Degrades silently per-feature: no light skill loaded → no dim mention;
    no timer skill loaded → no cooldown mention.

    bambu_monitor ALWAYS speaks its own "Print complete, sir — 'X' is
    finished." line on FINISH, so when it's loaded we must not repeat that
    head — we trail with just the offer (mirroring the milestone
    trail-behind), and return "" when there's nothing to offer so the user
    doesn't hear the same completion announced twice. Only when
    bambu_monitor is absent do we own the full announcement ourselves.
    """
    has_lights = _light_skill_available()
    has_timer  = _timer_skill_available()
    offers: list[str] = []
    if has_lights:
        offers.append("dim the workshop lights")
    if has_timer:
        offers.append("queue a cooldown timer")

    bambu_announces = _get_bambu_module() is not None
    if bambu_announces:
        if not offers:
            return ""
        return f"Shall I {' and '.join(offers)} for you, sir?"

    if pretty_filename:
        head = f"Print complete, sir — '{pretty_filename}' is finished."
    else:
        head = "Print complete, sir."
    if not offers:
        return head
    tail = f"Shall I {' and '.join(offers)} for you?"
    return f"{head} {tail}"


# ── patterns file ─────────────────────────────────────────────────
def _load_patterns() -> dict:
    if not os.path.exists(_PATTERNS_FILE):
        return {"buckets": {}}
    try:
        with open(_PATTERNS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("buckets"), dict):
            return d
    except Exception:
        pass
    return {"buckets": {}}


def _save_patterns(d: dict) -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        _atomic_write_json(_PATTERNS_FILE, d)
    except Exception as e:
        print(f"  [print_companion] patterns save failed: {e}")


def _record_print_outcome(outcome: str, state: dict) -> None:
    """Append a finished-print record to its bucket and save. Skips the
    write for prints we discovered mid-flight, since their start_ts is
    really `when JARVIS noticed` rather than `when printing began` and
    a wrong duration would poison future predictions."""
    if _print_midflight[0]:
        return
    fname    = state.get("filename") or _current_filename[0] or ""
    material = _infer_material(fname)
    total    = state.get("total_layer")
    key      = _bucket_key(material, total)
    started  = _print_start_ts[0] or None
    finished = time.time()
    duration_min = None
    if started and finished > started:
        duration_min = int(round((finished - started) / 60.0))
    record = {
        "filename":       fname,
        "material":       material,
        "layer_total":    total,
        "started_at":     started,
        "finished_at":    finished,
        "duration_min":   duration_min,
        "outcome":        outcome,
        "error_code":     state.get("print_error"),
        "vision_samples": list(_vision_samples),
    }
    data = _load_patterns()
    buckets = data.setdefault("buckets", {})
    series  = buckets.setdefault(key, [])
    series.append(record)
    if len(series) > PER_BUCKET_RETENTION:
        del series[: len(series) - PER_BUCKET_RETENTION]
    _save_patterns(data)


def _historical_failure_rate(material: str, total_layer) -> tuple:
    """Return (failure_rate, sample_count) for the bucket containing
    this material+layer combination."""
    key = _bucket_key(material, total_layer)
    data = _load_patterns()
    series = data.get("buckets", {}).get(key, [])
    if not series:
        return 0.0, 0
    fail = sum(1 for r in series if r.get("outcome") == "failed")
    return fail / len(series), len(series)


# ── vision ────────────────────────────────────────────────────────
def _vision_available() -> bool:
    """True only when local_vision is loaded and bobert_companion has
    its VLM hook wired up. Returns False on any uncertainty so we
    never crash on a probe."""
    if sys.modules.get("skill_local_vision") is None:
        return False
    bc = sys.modules.get("bobert_companion") or sys.modules.get("__main__")
    if bc is None:
        return False
    if not getattr(bc, "LOCAL_VISION_FALLBACK", False):
        return False
    if not getattr(bc, "LOCAL_VISION_MODEL", ""):
        return False
    return True


def _sample_vision_async(threshold: int) -> None:
    """Fire a best-effort VLM sample on a worker thread so the poll
    loop never blocks on Ollama latency. The excerpt is stashed in
    _vision_samples for inclusion in the eventual print record, and a
    proactive warning fires when the excerpt mentions a known failure
    signature."""
    if not _vision_available():
        return

    def _worker(_threshold=threshold):
        try:
            mod = sys.modules.get("skill_local_vision")
            fn = getattr(mod, "local_describe_screen", None) if mod else None
            if not callable(fn):
                return
            question = (
                "Look at the visible Bambu printer camera feed if any. "
                "Report any of: stringy extrusion, nozzle clog, adhesion "
                "loss, layer shift, blob, or detachment. Reply in one "
                "short sentence, or say 'nominal' if everything looks "
                "fine."
            )
            text = fn(question)
            if not isinstance(text, str) or not text.strip():
                return
            excerpt = text.strip()[:240]
            _vision_samples.append({
                "milestone": _threshold,
                "ts":        time.time(),
                "excerpt":   excerpt,
            })
            lower = excerpt.lower()
            if any(kw in lower for kw in _VISION_FAILURE_KEYWORDS):
                _enqueue_speech(
                    f"Sir, the camera feed is suggesting a potential "
                    f"failure signature at {_threshold}% — "
                    f"\"{excerpt[:120]}\". You may want to inspect."
                )
        except Exception as e:
            print(f"  [print_companion] vision sample failed: {e}")

    threading.Thread(target=_worker, daemon=True,
                     name="print_companion_vision").start()


# ── transition handlers ───────────────────────────────────────────
def _reset_per_print_state() -> None:
    _print_start_ts[0]              = time.time()
    _print_started_pct[0]           = None
    _print_midflight[0]             = False
    _saw_running_this_print[0]      = False
    _milestone_detected_at.clear()
    _milestone_announced.clear()
    _announced_completion_offer[0]  = False
    _warned_historical_failure[0]   = False
    _vision_samples.clear()


def _maybe_warn_historical_failure(state: dict) -> None:
    """At a new-print transition, peek at history for this bucket and
    warn the user when the failure rate is high enough to be useful."""
    if _warned_historical_failure[0]:
        return
    fname    = state.get("filename") or ""
    material = _infer_material(fname)
    total    = state.get("total_layer")
    rate, count = _historical_failure_rate(material, total)
    if count < MIN_HISTORY_FOR_WARNING:
        return
    if rate < HISTORICAL_FAILURE_WARN_THRESHOLD:
        return
    _warned_historical_failure[0] = True
    pct = int(round(rate * 100))
    try:
        layer_phrase = f"{int(total)}-layer" if total else "this size"
    except (TypeError, ValueError):
        layer_phrase = "this size"
    _enqueue_speech(
        f"Sir, a heads-up — past {material.upper()} prints around "
        f"{layer_phrase} have failed about {pct}% of the time across "
        f"{count} attempts. Worth keeping an eye on the first layer."
    )


def _maybe_announce_completion_offer(state: dict) -> None:
    if _announced_completion_offer[0]:
        return
    if not _saw_running_this_print[0]:
        # Never witnessed the print actually running — don't celebrate
        # a finish JARVIS only discovered post-hoc.
        return
    fname  = state.get("filename") or _current_filename[0] or ""
    pretty = _strip_filename(fname)
    line = _completion_offer_line(pretty)
    # Empty line means bambu_monitor already owns the completion
    # announcement and we have no offer to trail it with — stay quiet.
    if line:
        _enqueue_speech(line)
    _announced_completion_offer[0] = True
    _record_print_outcome("success", state)


def _maybe_handle_failed(state: dict) -> None:
    if _announced_completion_offer[0]:
        return
    if not _saw_running_this_print[0]:
        return
    _announced_completion_offer[0] = True
    _record_print_outcome("failed", state)


# ── poll loop ─────────────────────────────────────────────────────
def _poll_once() -> None:
    """Serialise the read-modify-write body under _poll_lock so the poll
    loop and bambu_monitor's MQTT-thread hook can't both fire a milestone
    or completion line by racing past the same not-yet-set guard."""
    with _poll_lock:
        _poll_once_locked()


def _poll_once_locked() -> None:
    state = _read_state()
    if state is None:
        return
    if state.get("last_update", 0.0) == 0.0:
        return

    gcode_state = (state.get("gcode_state") or "").upper()
    fname       = state.get("filename") or ""
    pct         = state.get("mc_percent")
    remaining   = state.get("mc_remaining")
    try:
        pct_f = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        pct_f = None

    prev_gcode = _last_gcode_state[0]
    _last_gcode_state[0] = gcode_state

    # New-print detection — either filename changed or gcode_state
    # rotated from a terminal back into RUNNING.
    new_print = False
    if fname and fname != _current_filename[0]:
        _current_filename[0] = fname
        new_print = True
    elif (prev_gcode is not None and gcode_state == "RUNNING"
          and prev_gcode in ("FINISH", "FAILED", "IDLE", "PREPARE")):
        new_print = True

    if new_print:
        _reset_per_print_state()
        _print_started_pct[0] = pct_f
        # Anything past the midflight threshold means we don't trust
        # our own start timestamp for history purposes.
        midflight = (pct_f is not None
                     and pct_f > MIDFLIGHT_PCT_THRESHOLD)
        # If this is the very first poll seeing any state, assume
        # mid-flight regardless — we don't know how long the printer
        # had been running before JARVIS came up.
        if prev_gcode is None:
            midflight = True
        _print_midflight[0] = midflight
        # Pre-mark already-crossed milestones so we don't blurt them
        # retroactively on a mid-flight discovery.
        if pct_f is not None:
            for threshold in (25, 50, 75):
                if pct_f >= threshold:
                    _milestone_announced.add(threshold)
        # Historical-failure heads-up only fires for genuine fresh
        # starts — printer state must be PREPARE or RUNNING and pct
        # must be near zero, otherwise the warning lands too late to
        # be useful.
        if (not midflight
            and gcode_state in ("RUNNING", "PREPARE")
            and (pct_f is None or pct_f < MIDFLIGHT_PCT_THRESHOLD)):
            _maybe_warn_historical_failure(state)
        return

    # No active print tracking yet (cold boot, idle printer).
    if _current_filename[0] is None:
        return

    if gcode_state == "RUNNING":
        _saw_running_this_print[0] = True

    # Milestone commentary — only for prints we caught from the start,
    # only while running, only once each.
    if (gcode_state == "RUNNING" and pct_f is not None
            and not _print_midflight[0]):
        now = time.time()
        for threshold in (25, 50, 75):
            if pct_f < threshold:
                continue
            if threshold in _milestone_announced:
                continue
            # Wait until bambu_monitor has already spoken its own
            # milestone line — we trail, never stack.
            if not _bambu_already_announced(threshold):
                continue
            first_seen = _milestone_detected_at.get(threshold)
            if first_seen is None:
                _milestone_detected_at[threshold] = now
                continue
            if now - first_seen < MILESTONE_OFFSET_SECONDS:
                continue
            _milestone_announced.add(threshold)
            _enqueue_speech(_milestone_commentary(threshold, remaining))
            _sample_vision_async(threshold)
            break  # one milestone per poll, never stack

    if gcode_state == "FINISH":
        _maybe_announce_completion_offer(state)
    elif gcode_state == "FAILED":
        _maybe_handle_failed(state)


def _poll_loop() -> None:
    if _stop_evt.wait(INITIAL_DELAY_SECONDS):
        return
    while not _stop_evt.is_set():
        try:
            _poll_once()
        except Exception:
            logging.exception("[print_companion] poll iteration crashed")
        if _stop_evt.wait(POLL_INTERVAL_SECONDS):
            return


def _start_poller() -> None:
    if _thread[0] is not None and _thread[0].is_alive():
        return
    _stop_evt.clear()
    t = threading.Thread(target=_poll_loop, daemon=True,
                         name="print_companion")
    t.start()
    _thread[0] = t


def stop_companion() -> None:
    """Tear down the poll thread — exposed for tests / hot-reload."""
    _stop_evt.set()
    _thread[0] = None


def _on_bambu_state_change(state: dict, prev_gcode, gcode_state: str) -> None:
    """Hook callback registered with bambu_monitor.register_state_change_hook.

    Lets the companion react in lockstep with bambu_monitor instead of
    waiting for the next 15s poll. The poll loop stays as a safety net for
    cold-start (state populated before the hook is registered) and for any
    transitions the hook somehow drops, but the hook path is what handles
    most state changes once both modules are loaded.
    """
    try:
        _poll_once()
    except Exception as e:
        print(f"  [print_companion] state-change hook crash: {e}")


def _register_bambu_hook() -> bool:
    """Best-effort wire-up into bambu_monitor's state-change hook. Returns
    True when registered, False when bambu_monitor isn't loaded or doesn't
    yet expose the hook API. The caller treats both False cases as 'fine'
    — the poll loop covers the gap."""
    bm = _get_bambu_module()
    if bm is None:
        return False
    register_fn = getattr(bm, "register_state_change_hook", None)
    if not callable(register_fn):
        return False
    try:
        register_fn(_on_bambu_state_change)
        return True
    except Exception as e:
        print(f"  [print_companion] bambu hook registration failed: {e}")
        return False


# ── actions ───────────────────────────────────────────────────────
def register(actions: dict) -> None:
    def print_companion_status(_: str = "") -> str:
        """Short status line — useful when the user asks 'is the
        companion watching the print?'."""
        state = _read_state()
        if state is None:
            return ("Print companion is loaded, sir, but the H2D "
                    "monitor isn't running.")
        if state.get("last_update", 0.0) == 0.0:
            return ("Print companion is armed, sir — no fresh printer "
                    "state yet.")
        gcode_state = (state.get("gcode_state") or "").upper() or "IDLE"
        fname       = _strip_filename(state.get("filename") or "")
        material    = _infer_material(state.get("filename") or "")
        total       = state.get("total_layer")
        rate, count = _historical_failure_rate(material, total)
        parts = [f"Companion tracking '{fname or 'no active print'}'"]
        parts.append(f"state {gcode_state}")
        if count >= MIN_HISTORY_FOR_WARNING:
            parts.append(
                f"history {int(round(rate * 100))}% failure across "
                f"{count} similar prints"
            )
        return ", ".join(parts) + ", sir."

    def print_companion_history(_: str = "") -> str:
        """Dump the per-bucket success/failure counts."""
        data = _load_patterns()
        buckets = data.get("buckets", {})
        if not buckets:
            return "No print history yet, sir."
        lines = []
        for key, records in sorted(buckets.items()):
            n = len(records)
            fails = sum(1 for r in records if r.get("outcome") == "failed")
            lines.append(f"  {key}: {n - fails} success / {fails} failed")
        return (f"Print companion history ({len(buckets)} bucket"
                f"{'s' if len(buckets) != 1 else ''}):\n" + "\n".join(lines))

    actions["print_companion_status"]  = print_companion_status
    actions["print_companion_history"] = print_companion_history

    # Hook into bambu_monitor's state-change dispatch so we react in the
    # same tick as the bambu announcements. If bambu_monitor isn't loaded
    # yet (skill load order, missing config) the poll loop below covers
    # state changes via its own _read_state() snapshot.
    _register_bambu_hook()

    _start_poller()
