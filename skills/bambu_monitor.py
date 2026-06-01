"""
Bambu H2D 3D-printer monitor skill for JARVIS.

Polls the printer's local MQTT API every POLL_INTERVAL_SECONDS while a print
is active and tracks:
  • current layer / total layers
  • estimated time remaining
  • error states (jam, AMS error, runout, etc.)
  • completion

Actions added:
  check_print           — verbally report current print status. Returns
                          'No active print' when idle, or 'Printing X,
                          layer 47 of 312, about 18 minutes remaining'.
  how_is_the_print      — same fields as check_print plus nozzle and bed
                          temperatures in a one-line JARVIS reply
                          ('Printing X, layer 47 of 312, about 18 minutes
                          remaining — nozzle at 220 degrees and bed at
                          60, sir.'). Alias: print_details.

Proactive speech (via bobert_companion.proactive_announce() → pending_speech.json):
  • 'Print started, sir — estimated 4 hours 12 minutes.' at 0 % (only when
    a print is observed transitioning out of PREPARE / FINISH / FAILED /
    IDLE, never for a print JARVIS discovered mid-flight on cold start).
  • 'Layer 1 adhesion looks nominal, sir.' once layer_num crosses past
    the first layer (suppressed for prints JARVIS picks up mid-flight).
  • 'Print is 25% complete, sir. Estimated 4 hours 12 minutes remaining.'
    at 25 %.
  • 'Print is 50% complete, sir. Estimated 2 hours 6 minutes remaining.'
    at 50 %.
  • 'Print is 75% complete, sir. Estimated 1 hour 3 minutes remaining.'
    at 75 %.
  • 'Print complete, sir. Shall I notify you when the bed has cooled?'
    on finish (100 %). Followed by 'The bed has cooled, sir — your part
    is ready to remove.' once bed_temper drops below 40 °C.
  • 'I'm afraid the print has failed at layer 47, sir...' on FAILED.
  • 'Slight problem, sir — your H2D appears to be unwell. Error code
    X on layer N.' on any non-zero in-flight print_error code (covers
    spaghetti detection, AMS errors, filament runout, etc.).
  • One-time 'Sir, I notice the H2D credentials aren't configured...' if
    BAMBU_PRINTER_IP isn't set on startup (tracked via flag file so it
    doesn't nag every launch).

Config in bobert_companion.py:
  BAMBU_PRINTER_IP    — printer's LAN IP (required; blank = skill is a no-op)
  BAMBU_ACCESS_CODE   — printer's LAN Access Code (required for MQTT)
  BAMBU_SERIAL        — printer's serial (required for MQTT topic)

If paho-mqtt isn't installed or any of the three config fields is blank, the
skill registers the check_print action but does not start the poller.
"""
import collections
import json
import logging
import os
import re
import threading
import time

from core.atomic_io import _atomic_write_json

try:
    import paho.mqtt.client as mqtt
    _HAS_MQTT = True
except ImportError:
    _HAS_MQTT = False

_PROJECT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE      = os.path.join(_PROJECT_DIR, "pending_speech.json")
_CREDS_PROMPT_FLAG = os.path.join(_PROJECT_DIR, "bambu_creds_prompted.flag")
# Sibling state file consumed by hud/bambu_h2d_overlay.py — atomic-written
# on every MQTT report so the overlay HUD picks up layer/percent/temp/error
# changes within one tick.
_OVERLAY_STATE_FILE = os.path.join(_PROJECT_DIR, "bambu_overlay_state.json")

POLL_INTERVAL_SECONDS = 60.0
INITIAL_DELAY_SECONDS = 30
COMPLETION_COOLDOWN   = 24 * 3600   # don't re-announce the same finish

# Once no MQTT push has landed for this long we treat the printer as
# offline/sleeping. is_printer_offline() returns True after that point so
# downstream callers (self_diagnostic._probe_bambu) can back off polling
# and stop logging chronic 5s connect timeouts.
OFFLINE_THRESHOLD_SECONDS = 300.0
# When offline, poll_loop sleeps at this interval instead of the normal
# 60s cadence so we're not nudging a dead session every minute.
OFFLINE_POLL_INTERVAL_SECONDS = 300.0

# Progress milestones to call out unprompted, in % complete order. Each entry
# is (threshold_pct, speech_template). Templates may include '{eta}' which is
# replaced with ' Estimated X remaining.' (or '' when ETA is unknown).
PROGRESS_MILESTONES = [
    (25, "Print is 25% complete, sir.{eta}"),
    (50, "Print is 50% complete, sir.{eta}"),
    (75, "Print is 75% complete, sir.{eta}"),
]

# Layer 1 adhesion is the failure-prone first layer; once layer_num reports a
# value at or past LAYER_ONE_ADHESION_LAYER we can safely tell sir the first
# layer made it down. (Bambu's layer_num is 1-indexed and reports the layer
# *currently* being printed, so layer_num >= 2 means layer 1 finished.)
LAYER_ONE_ADHESION_LAYER = 2

# After a print finishes JARVIS offers to call out when the bed has cooled.
# Below this temperature the printed part is safe to handle.
BED_COOL_THRESHOLD_C = 40.0

# Latest snapshot from MQTT push messages — refreshed in the on_message callback
_state_lock = threading.Lock()
_state: dict = {
    "stage":         None,   # int: printer state code
    "gcode_state":   None,   # str: IDLE / PREPARE / RUNNING / FINISH / FAILED / PAUSE
    "layer_num":     None,
    "total_layer":   None,
    "mc_percent":    None,
    "mc_remaining":  None,   # minutes
    "filename":      None,
    "print_error":   None,
    "nozzle_temper": None,   # °C, current nozzle temperature
    "bed_temper":    None,   # °C, current bed temperature
    "chamber_temper": None,  # °C, current chamber temperature (H2D enclosure)
    "ams_status":    None,   # raw AMS status payload (dict / int) when surfaced
    "last_update":   0.0,
}

# Last few chamber-temperature samples for swing detection. A short ring
# buffer (8 entries ≈ ~8 minutes at the 60s poll cadence) gives us enough
# signal to flag a meaningful swing without overreacting to a single
# anomalous reading.
_CHAMBER_HISTORY_LEN = 8
_chamber_history: collections.deque = collections.deque(maxlen=_CHAMBER_HISTORY_LEN)
# Chamber swing magnitude (°C) over the window that flips us from cyan
# (nominal) → amber (warning). The H2D enclosure should hold ±~3 °C while
# printing; a 6 °C swing is unusual enough to warrant a visual nudge.
_CHAMBER_SWING_AMBER_C = 6.0

_speech_lock = threading.Lock()
_last_finish_announced_at = [0.0]
_announced_error_codes: set = set()

# External listeners invoked after every state-change pass. Companion skills
# (e.g. proactive_print_companion) call register_state_change_hook() to be
# notified whenever _handle_state_change() runs. Hooks receive a single
# state-snapshot dict (read under _state_lock) plus prev/current gcode_state
# strings. Hook exceptions are swallowed so a buggy listener can't suppress
# bambu_monitor's own announcements.
_state_change_hooks: list = []
_state_change_hooks_lock = threading.Lock()

# task-71: persist the "I already announced this print finished" state so
# a JARVIS bounce while the printer is still showing FINISH doesn't replay
# the "Print complete" + "Bed has cooled" reminders on every boot.
_REMINDER_STATE_FILE = os.path.join(_PROJECT_DIR, "data", "bambu_reminder_state.json")
_reminder_state_lock = threading.Lock()


def _load_reminder_persistence() -> dict:
    with _reminder_state_lock:
        if not os.path.exists(_REMINDER_STATE_FILE):
            return {}
        try:
            with open(_REMINDER_STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except Exception:
            return {}


def _save_reminder_persistence(state: dict) -> None:
    with _reminder_state_lock:
        try:
            os.makedirs(os.path.dirname(_REMINDER_STATE_FILE), exist_ok=True)
            import tempfile
            d = os.path.dirname(_REMINDER_STATE_FILE)
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".bambu_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False)
                os.replace(tmp, _REMINDER_STATE_FILE)
            except Exception:
                try: os.remove(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  [bambu] reminder-state save failed: {e}")

# Module-level handles so bambu_setup.py can hot-restart the poller after
# the user runs the first-time setup wizard.
_mqtt_client = [None]
_poll_thread = [None]
_monitor_stop_evt = threading.Event()
# Set to True in the on_connect callback when paho reports rc=0, flipped back
# in on_disconnect. Drives is_printer_offline() — with connect_async() the
# client object exists immediately, so we can't infer reachability from its
# presence alone. Tracked as a one-element list so nested closures can mutate.
_mqtt_connected_ok: list = [False]

# Milestone bookkeeping — reset whenever a new print starts so the same
# print only announces each threshold once.
_announced_milestones: set      = set()
_current_print_filename: list   = [None]
_last_gcode_state: list         = [None]
_announced_start: list          = [False]   # 0 % "Print started" guard
_announced_layer1: list         = [False]   # "Layer 1 adhesion looks nominal" guard
_post_finish_bed_watch: list    = [False]   # armed when we announce FINISH; cleared on cool-down call-out
_bed_cool_announced: list       = [False]
# Captured at the 0 % "Print started" announcement so predictive_morning_setup
# can report a finish time and "N hours under/over estimate" for the most
# recent overnight print. Cleared on each new_print transition.
_print_start_ts: list           = [0.0]
_print_initial_estimate_min: list = [None]


def _enqueue_speech(message: str) -> None:
    """Route a proactive announcement through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json if the parent module hasn't loaded yet
    (e.g. unit test, import-time skill registration before bobert_companion
    finishes initialising)."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="bambu")
            return
    except Exception:
        # Fall through to local write — never let a broken parent import
        # silence a print-status alert.
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
            # Atomic write failed (e.g. read-only network share, full disk,
            # permission denied). Fall back to console so the alert isn't
            # silently lost — at minimum the user sees it in the log stream.
            print(f"  [bambu] speech-queue write failed ({e}); alert: {message}")


def _maybe_prompt_for_credentials() -> None:
    """Speak a one-time JARVIS-style nudge when BAMBU_PRINTER_IP is missing.

    Uses a flag file so this only fires on the very first launch after the IP
    is cleared — won't pester the user every restart while they're not using
    the printer. Delete `bambu_creds_prompted.flag` to re-arm it.
    """
    if os.path.exists(_CREDS_PROMPT_FLAG):
        return
    _enqueue_speech(
        "Sir, I notice the H2D credentials aren't configured. "
        "Shall I walk you through it?"
    )
    try:
        with open(_CREDS_PROMPT_FLAG, "w", encoding="utf-8") as f:
            f.write(f"prompted at {time.time()}\n")
    except Exception as e:
        # If the flag write fails the prompt will repeat next launch — that's
        # the more user-visible failure mode, so just log and move on.
        print(f"  [bambu] could not write creds-prompt flag: {e}")


def _read_config() -> tuple[str, str, str]:
    """Pull printer config from bobert_companion at call time."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        ip      = getattr(bc, "BAMBU_PRINTER_IP",   "") or ""
        access  = getattr(bc, "BAMBU_ACCESS_CODE",  "") or ""
        serial  = getattr(bc, "BAMBU_SERIAL",       "") or ""
        return ip.strip(), access.strip(), serial.strip()
    except Exception:
        return "", "", ""


def _format_minutes(minutes) -> str:
    """Render minute count as 'X hours Y minutes' for speech."""
    try:
        m = int(minutes)
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return "less than a minute"
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    hours = m // 60
    rem = m % 60
    if rem == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours} hour{'s' if hours != 1 else ''} and {rem} minute{'s' if rem != 1 else ''}"


def _format_temp(t) -> str:
    """Render a temperature reading as 'N degrees' for speech.

    Bambu reports temps as floats (e.g. 219.8); we round to the nearest
    whole degree because half-degree precision is meaningless out loud.
    Returns '' when the field is missing or non-numeric so callers can
    skip the line entirely.
    """
    try:
        v = float(t)
    except (TypeError, ValueError):
        return ""
    if v < 1:
        return ""  # 0 °C readings usually mean "sensor not active"
    return f"{int(round(v))} degrees"


def _strip_filename(name: str) -> str:
    """Make a printer filename pronounceable."""
    if not name:
        return ""
    base = os.path.basename(name)
    base = re.sub(r"\.(3mf|gcode|gco)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_\-]+", " ", base).strip()
    return base[:60] if base else ""


def _compute_risk_level() -> tuple[int, str]:
    """Classify current print health for the HUD overlay.

    Returns (risk_level, note) where:
       0 → nominal (cyan)
       1 → warning (amber) — chamber temp swings beyond _CHAMBER_SWING_AMBER_C,
           or AMS reporting a non-fatal anomaly we can't classify further
       2 → failure-risk (red) — print_error non-zero, FAILED state, or
           AMS surfacing a hard fault keyword
    Caller holds no lock — this function takes _state_lock briefly itself.

    Notes are short phrases (≤30 chars) so they fit the overlay footer
    without truncation.
    """
    with _state_lock:
        gcode_state = (_state.get("gcode_state") or "").upper()
        err         = _state.get("print_error")
        ams         = _state.get("ams_status")
        # Snapshot the history under the lock to avoid races with on_message.
        history = list(_chamber_history)

    # Hard fault: explicit FAILED or any non-zero print_error.
    if gcode_state == "FAILED":
        return 2, "FAILED — check printer"
    if err and err not in (0, "0", None, ""):
        return 2, f"Error code {err}"

    # AMS hard-fault keywords. Bambu surfaces AMS state in several shapes:
    # an int bitmask, an object with a 'humidity'/'tray'/'state', or a
    # free-text string. We scan the JSON-encoded form for the obvious
    # failure words so we don't have to parse every firmware revision.
    if ams is not None:
        try:
            ams_str = json.dumps(ams).lower() if not isinstance(ams, str) else ams.lower()
        except Exception:
            ams_str = ""
        if any(kw in ams_str for kw in ("jam", "runout", "fault", "error")):
            return 2, "AMS fault — check spools"

    # Chamber temperature swing — needs at least 3 samples to be meaningful.
    if len(history) >= 3:
        swing = max(history) - min(history)
        if swing >= _CHAMBER_SWING_AMBER_C:
            return 1, f"Chamber swing {swing:.1f}°C"

    return 0, ""


def _write_overlay_state() -> None:
    """Atomic-write a compact snapshot for hud/bambu_h2d_overlay.py.

    Always writes even when the printer is idle so the overlay can read
    `gcode_state` and dismiss itself cleanly — the overlay-watcher in
    holographic_overlay.py keys off `gcode_state` to decide when to spawn
    or shut down the widget.
    """
    risk, note = _compute_risk_level()
    with _state_lock:
        snapshot = {
            "gcode_state":   _state.get("gcode_state"),
            "stage":         _state.get("stage"),
            "filename":      _strip_filename(_state.get("filename") or ""),
            "layer_num":     _state.get("layer_num"),
            "total_layer":   _state.get("total_layer"),
            "mc_percent":    _state.get("mc_percent"),
            "mc_remaining":  _state.get("mc_remaining"),
            "nozzle_temper": _state.get("nozzle_temper"),
            "bed_temper":    _state.get("bed_temper"),
            "chamber_temper": _state.get("chamber_temper"),
            "print_error":   _state.get("print_error"),
            "risk_level":    risk,
            "risk_note":     note,
            "last_update":   _state.get("last_update", 0.0),
            "written_at":    time.time(),
        }
    try:
        # Route through the shared helper so we inherit its Windows
        # PermissionError retry — the overlay reader polls this file at
        # ~1Hz and used to race os.replace into WinError 5 on every
        # cycle that overlapped a read.
        _atomic_write_json(_OVERLAY_STATE_FILE, snapshot, indent=None)
    except Exception as e:
        # Overlay state is best-effort; a failure here shouldn't take
        # down the MQTT callback.
        print(f"  [bambu] overlay state write failed: {e}")


def _on_message(client, userdata, msg) -> None:
    """MQTT callback — Bambu pushes a 'report' object. Extract the print fields
    we care about into _state."""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return
    report = payload.get("print", payload)

    with _state_lock:
        _state["last_update"] = time.time()
        for key in ("stage", "gcode_state", "layer_num", "total_layer",
                    "mc_percent", "mc_remaining", "print_error",
                    "nozzle_temper", "bed_temper", "chamber_temper"):
            if key in report:
                _state[key] = report[key]
        # AMS status keys vary by firmware revision — accept either the
        # 'ams' object or a flat 'ams_status' / 'ams_exist_bits' field.
        for akey in ("ams", "ams_status", "ams_exist_bits"):
            if akey in report:
                _state["ams_status"] = report[akey]
                break
        # File name lives under different keys depending on firmware
        for fkey in ("subtask_name", "gcode_file", "subtask"):
            if report.get(fkey):
                _state["filename"] = report[fkey]
                break
        # Track chamber temperature samples for swing detection. Only
        # record genuine readings (>1 °C) so a missing-sensor zero doesn't
        # widen the apparent swing.
        chamber = _state.get("chamber_temper")
        try:
            if chamber is not None and float(chamber) > 1.0:
                _chamber_history.append(float(chamber))
        except (TypeError, ValueError):
            pass

    _handle_state_change()
    _write_overlay_state()


def register_state_change_hook(callback) -> None:
    """Register a callback fired after every state-change pass.

    Callback signature: ``callback(state_snapshot: dict, prev_gcode: str|None,
    gcode_state: str) -> None``. The snapshot is a shallow copy of `_state`
    taken under `_state_lock`, so listeners can read it without acquiring the
    lock themselves. Exceptions raised by hooks are caught and printed —
    bambu_monitor must never let a buggy listener silence its own callouts.
    """
    with _state_change_hooks_lock:
        if callback not in _state_change_hooks:
            _state_change_hooks.append(callback)


def _fire_state_change_hooks(prev_gcode, gcode_state: str) -> None:
    with _state_change_hooks_lock:
        hooks = list(_state_change_hooks)
    if not hooks:
        return
    with _state_lock:
        snapshot = dict(_state)
    for fn in hooks:
        try:
            fn(snapshot, prev_gcode, gcode_state)
        except Exception as e:
            print(f"  [bambu] state-change hook {getattr(fn, '__name__', '?')} "
                  f"raised: {e}")


def _handle_state_change() -> None:
    """Inspect _state, queue spoken alerts for milestone / finish / error
    transitions, then dispatch to any registered state-change hooks so
    companion modules (e.g. proactive_print_companion) can react in lockstep
    with the rest of the announcement pipeline.
    """
    with _state_lock:
        gcode_state = (_state.get("gcode_state") or "").upper()
        err         = _state.get("print_error")
        layer       = _state.get("layer_num")
        percent     = _state.get("mc_percent")
        remaining   = _state.get("mc_remaining")
        fname       = _strip_filename(_state.get("filename") or "")
        bed         = _state.get("bed_temper")

    # Detect new-print transitions: filename change, OR moving back into
    # RUNNING from a terminal/idle state. Either resets milestone + error
    # bookkeeping so the next print can announce its own progress.
    prev_gcode = _last_gcode_state[0]
    _last_gcode_state[0] = gcode_state

    new_print = False
    if fname and fname != _current_print_filename[0]:
        _current_print_filename[0] = fname
        new_print = True
    elif gcode_state == "RUNNING" and prev_gcode in ("FINISH", "FAILED", "IDLE", "PREPARE"):
        new_print = True

    if new_print:
        _announced_milestones.clear()
        _announced_error_codes.clear()
        _last_finish_announced_at[0] = 0.0
        _announced_start[0] = False
        _announced_layer1[0] = False
        _post_finish_bed_watch[0] = False
        _bed_cool_announced[0] = False
        _print_start_ts[0] = 0.0
        _print_initial_estimate_min[0] = None

    # Coerce percent to a float once so all comparisons agree.
    try:
        pct = float(percent) if percent is not None else None
    except (TypeError, ValueError):
        pct = None

    # FIX (2026-05-30): whenever milestone tracking (re)starts — a genuine new
    # print OR a spurious RUNNING <-> PREPARE/IDLE gcode flicker (or filename
    # jitter) on a print we're already mid-way through — pre-mark every
    # milestone the print has ALREADY passed. Without this, a flicker cleared
    # _announced_milestones and the very next tick announced "Sir, the print is
    # 10% complete" even though the print was really at 41 % — the exact
    # wrong/repeated readout the user called out ("told me 40 times and it's
    # wrong"). A genuinely fresh print sits near 0 %, so nothing is pre-marked
    # and real milestone crossings still announce normally.
    if new_print and pct is not None:
        for _threshold, _t in PROGRESS_MILESTONES:
            if pct >= _threshold:
                _announced_milestones.add(_threshold)

    # Cold start: if we're observing a print already past one or more
    # milestones, mark those as "done" so we don't blurt 25 / 50 / 75 in a
    # row when JARVIS restarts mid-print. Also suppress the 0 % start
    # announcement — we didn't actually witness the print start.
    if prev_gcode is None and gcode_state == "RUNNING":
        _announced_start[0] = True
        if pct is not None:
            for threshold, _template in PROGRESS_MILESTONES:
                if pct >= threshold:
                    _announced_milestones.add(threshold)
        try:
            if layer is not None and int(layer) >= LAYER_ONE_ADHESION_LAYER:
                _announced_layer1[0] = True
        except (TypeError, ValueError):
            pass

    # 0 % start announcement — fire once when we genuinely catch a new
    # print transitioning into RUNNING. The new_print guard ensures we
    # only fire after the bookkeeping reset above; the prev_gcode check
    # (set to True in the cold-start branch) keeps us quiet for prints
    # discovered mid-flight.
    if gcode_state == "RUNNING" and not _announced_start[0]:
        _announced_start[0] = True
        _print_start_ts[0] = time.time()
        try:
            _print_initial_estimate_min[0] = int(remaining) if remaining else None
        except (TypeError, ValueError):
            _print_initial_estimate_min[0] = None
        remaining_str = _format_minutes(remaining) if remaining else ""
        if remaining_str:
            _enqueue_speech(f"Print started, sir — estimated {remaining_str}.")
        else:
            _enqueue_speech("Print started, sir.")

    # Progress milestones (only while running). Each milestone announcement
    # is sent through proactive_announce() (via _enqueue_speech) so the main
    # listen loop will speak it at the next turn boundary.
    if gcode_state == "RUNNING" and pct is not None:
        for threshold, template in PROGRESS_MILESTONES:
            if pct >= threshold and threshold not in _announced_milestones:
                _announced_milestones.add(threshold)
                remaining_str = _format_minutes(remaining) if remaining else ""
                eta_tail = f" Estimated {remaining_str} remaining." if remaining_str else ""
                _enqueue_speech(template.format(eta=eta_tail))
                break  # one milestone per push — never stack announcements

    # Layer 1 adhesion check. Once we observe layer_num past the first layer
    # we can credibly report that adhesion held. Fires once per print; the
    # cold-start branch above pre-arms this for prints discovered mid-flight
    # so we never blurt a layer-1 announcement for a print already at layer
    # 200.
    if gcode_state == "RUNNING" and not _announced_layer1[0]:
        try:
            if layer is not None and int(layer) >= LAYER_ONE_ADHESION_LAYER:
                _announced_layer1[0] = True
                _enqueue_speech("Layer 1 adhesion looks nominal, sir.")
        except (TypeError, ValueError):
            pass

    # Completion (100 %)
    if gcode_state == "FINISH":
        now = time.time()
        # task-71: durable check — has THIS specific print's FINISH already
        # been announced? Keyed by filename so a NEW print after a bounce
        # still announces, but the same print doesn't replay.
        rstate = _load_reminder_persistence()
        finish_key = f"finish:{fname or '_anon_'}"
        already_announced = bool(rstate.get(finish_key))
        time_gate_ok = (now - _last_finish_announced_at[0]) > COMPLETION_COOLDOWN
        if time_gate_ok and not already_announced:
            _last_finish_announced_at[0] = now
            if fname:
                _enqueue_speech(
                    f"Print complete, sir — '{fname}' is finished. "
                    "Shall I notify you when the bed has cooled?"
                )
            else:
                _enqueue_speech(
                    "Print complete, sir. "
                    "Shall I notify you when the bed has cooled?"
                )
            _post_finish_bed_watch[0] = True
            _bed_cool_announced[0] = False
            # Persist so a bounce doesn't replay this same print's FINISH.
            rstate[finish_key] = {"ts": now, "fname": fname}
            # Prune entries older than 7 days — keeps the file from growing.
            cutoff = now - 7 * 86400
            rstate = {k: v for k, v in rstate.items()
                      if not (isinstance(v, dict) and v.get("ts", 0) < cutoff)}
            _save_reminder_persistence(rstate)
        elif already_announced:
            # Quietly arm the bed-watch without re-announcing — so the bed-
            # cool follow-up still fires if it hasn't yet for this print.
            bed_key = f"bedcool:{fname or '_anon_'}"
            if not rstate.get(bed_key):
                _post_finish_bed_watch[0] = True
                _bed_cool_announced[0] = False

    # Bed cool-down follow-up. Only armed after we just announced a FINISH,
    # so a print JARVIS discovered already-finished on cold start (and which
    # never produced the "Shall I notify you..." offer) never produces a
    # phantom cool-down call-out either.
    if _post_finish_bed_watch[0] and not _bed_cool_announced[0]:
        try:
            if bed is not None and float(bed) < BED_COOL_THRESHOLD_C:
                rstate = _load_reminder_persistence()
                bed_key = f"bedcool:{fname or '_anon_'}"
                if not rstate.get(bed_key):
                    _bed_cool_announced[0] = True
                    _post_finish_bed_watch[0] = False
                    _enqueue_speech(
                        "The bed has cooled, sir — your part is ready to remove."
                    )
                    rstate[bed_key] = {"ts": time.time(), "fname": fname}
                    _save_reminder_persistence(rstate)
                else:
                    # Already told the user. Quietly clear flags.
                    _bed_cool_announced[0] = True
                    _post_finish_bed_watch[0] = False
        except (TypeError, ValueError):
            pass

    # In-flight errors — Bambu surfaces spaghetti detection, AMS faults,
    # filament runout, and similar recoverable problems through print_error
    # while gcode_state is still RUNNING / PAUSE. Announce each distinct
    # code once per print so JARVIS interrupts immediately but doesn't loop.
    if err and err not in (0, "0", None):
        key = str(err)
        if key not in _announced_error_codes:
            # Durable per-(print, code) dedup — like FINISH — so a bounce or a
            # RUNNING<->PAUSE flicker can't re-announce the same error. This is
            # the "told me 40 times" complaint: the in-memory set alone reset on
            # every restart and re-blurted the same 'error on layer 142'.
            rstate = _load_reminder_persistence()
            err_key = f"err:{fname or '_anon_'}:{key}"
            _announced_error_codes.add(key)
            if not rstate.get(err_key):
                layer_str = f" on layer {layer}" if layer else ""
                _enqueue_speech(
                    f"Slight problem, sir — your H2D appears to be unwell. "
                    f"Error code {key}{layer_str}."
                )
                rstate[err_key] = {"ts": time.time(), "fname": fname}
                _save_reminder_persistence(rstate)

    # Failed state — surface the layer so the user knows roughly when it died
    if gcode_state == "FAILED":
        key = "failed"
        if key not in _announced_error_codes:
            _announced_error_codes.add(key)
            layer_str = f" at layer {layer}" if layer else ""
            _enqueue_speech(
                f"I'm afraid the print has failed{layer_str}, sir. "
                "You'll want to check the printer."
            )

    # Fan out to companion modules (proactive_print_companion, etc.).
    # Hook errors are swallowed inside _fire_state_change_hooks so a buggy
    # listener can never silence the alerts above.
    _fire_state_change_hooks(prev_gcode, gcode_state)


# Backwards-compatible alias. Older callers (tests, hot-reload code,
# proactive_print_companion versions that pre-date the rename) still reference
# `_check_for_announcements`; keep both names bound to the same function so
# nothing breaks.
_check_for_announcements = _handle_state_change


def _start_mqtt(ip: str, access: str, serial: str):
    """Stage a non-blocking MQTT connection and start the network loop.

    Uses ``connect_async()`` so the boot thread isn't blocked by a TCP
    handshake when the printer is powered down — paho's loop thread does
    the actual connect attempt and any subsequent reconnects in the
    background. The reconnect delay is stretched so a chronically offline
    printer doesn't generate a 5s connect timeout every minute.
    """
    if not _HAS_MQTT:
        print("  [bambu] paho-mqtt not installed — skip start. Run: pip install paho-mqtt")
        return None

    topic = f"device/{serial}/report"

    def _on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            _mqtt_connected_ok[0] = True
            print(f"  [bambu] MQTT connected, subscribing to {topic}")
            client.subscribe(topic)
        else:
            _mqtt_connected_ok[0] = False
            print(f"  [bambu] MQTT connect failed rc={rc}")

    def _on_disconnect(client, userdata, *args, **kwargs):
        # paho 1.x signature: (client, userdata, rc). 2.x adds flags+properties.
        # We don't care about rc here — just flip the reachability flag so
        # is_printer_offline() can back off downstream probes.
        _mqtt_connected_ok[0] = False

    # paho-mqtt 2.x requires callback_api_version as a kwarg; 1.x rejects it.
    _client_kwargs = {
        "client_id": f"jarvis-{int(time.time())}",
        "protocol":  mqtt.MQTTv311,
    }
    if hasattr(mqtt, "CallbackAPIVersion"):
        _client_kwargs["callback_api_version"] = mqtt.CallbackAPIVersion.VERSION2
    client = mqtt.Client(**_client_kwargs)
    client.username_pw_set("bblp", access)
    client.tls_set(cert_reqs=mqtt.ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect
    client.on_message = _on_message
    # Throttle paho's automatic reconnects. With a chronically offline printer
    # the default 1–120s exponential backoff settles at 120s, still causing a
    # ~30s connect-timeout cascade in the diag logs. 60s min / 600s max keeps
    # us courteous to a sleeping printer and gives is_printer_offline() room
    # to skip self_diagnostic probes between attempts.
    try:
        client.reconnect_delay_set(min_delay=60, max_delay=600)
    except Exception:
        pass
    try:
        client.connect_async(ip, 8883, keepalive=60)
        client.loop_start()
        return client
    except Exception as e:
        # connect_async normally only raises on bad arguments (e.g. bogus
        # hostname), not on network failure — those surface via on_connect
        # with a non-zero rc. Treat any raise here as a hard failure.
        print(f"  [bambu] could not schedule connect to {ip}:8883 — {e}")
        return None


def _poll_loop(client, stop_evt: threading.Event) -> None:
    """Periodically request a fresh push (Bambu pushes updates on its own
    but we nudge it every minute to keep state warm and detect stalls).

    Honours `stop_evt` so a setup wizard can swap in a freshly-configured
    client without leaving the old thread looping against a dead session.
    """
    if stop_evt.wait(INITIAL_DELAY_SECONDS):
        return
    while not stop_evt.is_set():
        try:
            with _state_lock:
                age = time.time() - _state["last_update"]
            if age > 300:
                # Re-trigger by sending a 'pushall' request (Bambu's command
                # to dump full state). Topic is device/<serial>/request.
                try:
                    config_ip, config_access, config_serial = _read_config()
                    if config_serial:
                        topic = f"device/{config_serial}/request"
                        payload = json.dumps({
                            "pushing": {"sequence_id": "0", "command": "pushall"}
                        })
                        client.publish(topic, payload)
                except Exception as e:
                    print(f"  [bambu] pushall failed: {e}")
        except Exception:
            logging.exception("[bambu] poll loop iteration crashed")
        # When the printer's been silent past the offline threshold, stretch
        # the poll interval to 5 minutes so we're not nudging a dead session
        # every 60s. A fresh MQTT message in _on_message resets last_update
        # and the next iteration drops back to the normal cadence.
        interval = (OFFLINE_POLL_INTERVAL_SECONDS
                    if is_printer_offline()
                    else POLL_INTERVAL_SECONDS)
        if stop_evt.wait(interval):
            return


def is_printer_offline() -> bool:
    """True when the printer is unreachable / asleep and pollers should back off.

    Treated as online when we either have a fresh MQTT push (within
    OFFLINE_THRESHOLD_SECONDS) or paho currently believes it holds a session
    (on_connect fired with rc=0 and on_disconnect hasn't been called since).
    Otherwise the printer is presumed offline.

    During the early-boot window we report offline even when the connection
    state is unknown — connect_async() is non-blocking, so the first
    on_connect callback can take a few seconds to arrive, and downstream
    probes should hold off rather than fire a 5s timeout in that gap.

    Returns False when the skill is unconfigured.
    """
    ip, access, serial = _read_config()
    if not (ip and access and serial):
        return False  # unconfigured — not "offline", just not active
    with _state_lock:
        last_update = _state.get("last_update", 0.0)
    now = time.time()
    if last_update and (now - last_update) <= OFFLINE_THRESHOLD_SECONDS:
        return False
    if _mqtt_connected_ok[0]:
        # paho says the session is up. No recent push, but the printer is
        # reachable — don't claim offline.
        return False
    return True


def stop_monitor() -> None:
    """Tear down the running MQTT client + poll thread so a fresh start
    after credential changes doesn't leave a zombie session behind."""
    _monitor_stop_evt.set()
    client = _mqtt_client[0]
    if client is not None:
        try:
            # loop_stop() tears down paho's network thread, which also
            # interrupts any in-flight connect_async() attempt. disconnect()
            # follows so the printer (if reachable) sees a clean close.
            client.loop_stop()
            client.disconnect()
        except Exception as e:
            print(f"  [bambu] disconnect on stop failed: {e}")
    _mqtt_client[0] = None
    _poll_thread[0] = None
    _mqtt_connected_ok[0] = False
    # Reset state so a stale snapshot from the old session doesn't fool
    # check_print into reporting against the wrong printer.
    with _state_lock:
        for k in list(_state.keys()):
            _state[k] = None if k != "last_update" else 0.0
        _chamber_history.clear()
    # Mirror the reset into the overlay snapshot so the HUD watcher
    # promptly retires the widget instead of staring at stale layer
    # numbers from the prior session.
    _write_overlay_state()


def start_monitor() -> bool:
    """Read config, spin up the MQTT client + poll thread. Idempotent:
    if a monitor is already running it's torn down first. Returns True
    when polling is active, False if config is missing or paho-mqtt
    isn't installed."""
    if _mqtt_client[0] is not None or _poll_thread[0] is not None:
        stop_monitor()
    ip, access, serial = _read_config()
    if not ip or not access or not serial:
        print("  [bambu] BAMBU_PRINTER_IP / ACCESS_CODE / SERIAL not "
              "configured — poller disabled")
        return False
    if not _HAS_MQTT:
        print("  [bambu] paho-mqtt not installed — poller disabled. "
              "pip install paho-mqtt")
        return False
    client = _start_mqtt(ip, access, serial)
    if client is None:
        return False
    _monitor_stop_evt.clear()
    t = threading.Thread(target=_poll_loop, args=(client, _monitor_stop_evt),
                         daemon=True)
    t.start()
    _mqtt_client[0] = client
    _poll_thread[0] = t
    print(f"  [bambu] monitor active — polling {ip} every "
          f"{POLL_INTERVAL_SECONDS:.0f}s")
    return True


def get_last_print_completion_summary(within_seconds: float = 12 * 3600) -> dict | None:
    """Return a small dict describing the most recently FINISHED print, or
    None if nothing has finished within `within_seconds` (default 12h —
    "overnight" window for predictive_morning_setup).

    Keys (all best-effort; absent if not available):
        finish_ts        — epoch seconds when JARVIS announced FINISH
        finish_phrase    — '4:12 AM' style local-time phrase
        filename         — stripped gcode filename, '' if unknown
        elapsed_minutes  — actual print duration in minutes (None if start
                           wasn't observed)
        estimated_minutes — initial estimate captured at the 0% start
                           announcement (None if not observed)
        delta_minutes    — estimated_minutes - elapsed_minutes; positive
                           means finished UNDER estimate. None if either
                           input is missing.
    """
    finish_ts = _last_finish_announced_at[0]
    if not finish_ts or (time.time() - finish_ts) > within_seconds:
        return None
    with _state_lock:
        state = (_state.get("gcode_state") or "").upper()
        fname = _strip_filename(_state.get("filename") or "")
    if state != "FINISH":
        # State has moved on (IDLE / new print) since the finish — the
        # finish_ts is still authoritative but we can't trust the filename.
        fname = ""
    finish_lt = time.localtime(finish_ts)
    disp_hour = finish_lt.tm_hour % 12 or 12
    ampm = "AM" if finish_lt.tm_hour < 12 else "PM"
    finish_phrase = f"{disp_hour}:{finish_lt.tm_min:02d} {ampm}"

    start_ts = _print_start_ts[0]
    elapsed_minutes = None
    if start_ts and start_ts < finish_ts:
        elapsed_minutes = int(round((finish_ts - start_ts) / 60.0))

    estimated_minutes = _print_initial_estimate_min[0]
    delta_minutes = None
    if elapsed_minutes is not None and estimated_minutes is not None:
        delta_minutes = int(estimated_minutes - elapsed_minutes)

    return {
        "finish_ts":        finish_ts,
        "finish_phrase":    finish_phrase,
        "filename":         fname,
        "elapsed_minutes":  elapsed_minutes,
        "estimated_minutes": estimated_minutes,
        "delta_minutes":    delta_minutes,
    }


def register(actions):
    def check_print(_: str = "") -> str:
        with _state_lock:
            gcode_state = (_state.get("gcode_state") or "").upper()
            layer       = _state.get("layer_num")
            total       = _state.get("total_layer")
            remaining   = _state.get("mc_remaining")
            fname       = _strip_filename(_state.get("filename") or "")
            last_update = _state.get("last_update", 0.0)

        if last_update == 0.0:
            return ("I don't have a fresh status from the printer yet, sir. "
                    "Either it isn't reachable or the monitor hasn't connected.")

        if gcode_state in ("IDLE", "", None) and not layer:
            return "No active print, sir. The printer is idle."

        if gcode_state == "FINISH":
            who = f" of '{fname}'" if fname else ""
            return f"The print{who} has finished, sir."

        if gcode_state == "PAUSE":
            return "The print is currently paused, sir."

        parts = []
        if fname:
            parts.append(f"Printing '{fname}'")
        else:
            parts.append("Print in progress")
        if layer and total:
            parts.append(f"layer {layer} of {total}")
        remaining_str = _format_minutes(remaining) if remaining else ""
        if remaining_str:
            parts.append(f"about {remaining_str} remaining")
        return ", ".join(parts) + ", sir."

    def how_is_the_print(_: str = "") -> str:
        """Return ETA + current layer + nozzle/bed temps as a one-line
        JARVIS-style reply. Triggered by phrases like 'how's the print'.
        """
        with _state_lock:
            gcode_state    = (_state.get("gcode_state") or "").upper()
            layer          = _state.get("layer_num")
            total          = _state.get("total_layer")
            remaining      = _state.get("mc_remaining")
            fname          = _strip_filename(_state.get("filename") or "")
            nozzle         = _state.get("nozzle_temper")
            bed            = _state.get("bed_temper")
            last_update    = _state.get("last_update", 0.0)

        if last_update == 0.0:
            return ("I don't have a fresh status from the printer yet, sir. "
                    "Either it isn't reachable or the monitor hasn't connected.")

        if gcode_state in ("IDLE", "", None) and not layer:
            return "No active print, sir. The printer is idle."

        if gcode_state == "FINISH":
            who = f" of '{fname}'" if fname else ""
            return f"The print{who} has finished, sir."

        nozzle_str = _format_temp(nozzle)
        bed_str    = _format_temp(bed)
        temp_parts = []
        if nozzle_str:
            temp_parts.append(f"nozzle at {nozzle_str}")
        if bed_str:
            temp_parts.append(f"bed at {bed_str}")
        temp_tail = (" — " + " and ".join(temp_parts)) if temp_parts else ""

        if gcode_state == "PAUSE":
            return f"The print is currently paused, sir{temp_tail}."

        parts = []
        if fname:
            parts.append(f"Printing '{fname}'")
        else:
            parts.append("Print in progress")
        if layer and total:
            parts.append(f"layer {layer} of {total}")
        remaining_str = _format_minutes(remaining) if remaining else ""
        if remaining_str:
            parts.append(f"about {remaining_str} remaining")
        return ", ".join(parts) + f"{temp_tail}, sir."

    actions["check_print"]      = check_print
    actions["how_is_the_print"] = how_is_the_print
    actions["print_details"]    = how_is_the_print  # alias

    ip, _access, _serial = _read_config()
    if not ip:
        _maybe_prompt_for_credentials()
    start_monitor()
