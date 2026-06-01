"""
Bambu H2D voice companion — in-character announcements for H2D prints.

The user owns a Bambu H2D and there's no current skill that speaks print
progress aloud with the JARVIS voice signature ("sir", "I'm afraid...",
"slight problem"). This module rides on top of bambu_monitor's state-change
dispatch and adds:

  • Milestone announcements at 25 / 50 / 75 % phrased as
    "Print at 25%, sir — roughly 2 hours 14 minutes remaining."
    (distinct from bambu_monitor's terser "Print is 25% complete" line and
    proactive_print_companion's MCU-flavoured "Quarter of the way along"
    trailing commentary).
  • Error / FAILED announcements phrased in the JARVIS register —
    "Layer shift detected on plate 1, I'm afraid we have a problem.",
    "I'm afraid the AMS appears to be unwell, sir."
  • A `print_status` action — discoverable alias-style entry point that
    returns the same field set as bambu_monitor.how_is_the_print
    (filename, layer/total, ETA, nozzle + bed temps) so users who ask
    "what's the print status?" land on the right action.

Double-speech avoidance
-----------------------
bambu_monitor (25/50/75% terse), bambu_print_announcer (10/95% gated),
and proactive_print_companion (25/50/75% trailing flavour) already speak
at these thresholds. To avoid stacking, this skill routes its milestone
announcements through bambu_print_announcer._proactive_announce when
available — that path is gated by dnd_focus_mode and rate-limited to one
announcement per 10 minutes across both companion skills. Error / FAILED
callouts bypass the rate-limit (via direct queue) because a layer shift
shouldn't be silenced by a chatty 50% milestone.

Graceful degradation
--------------------
register_state_change_hook() is the only bambu_monitor entry point we
depend on. When bambu_monitor isn't loaded:
  • the hook never registers (the call is wrapped in a try block)
  • print_status returns an "offline" message instead of crashing
  • no announcements fire — the skill is effectively dormant
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

# Milestone thresholds we voice. Bambu_monitor already speaks plainer
# lines here, but our phrasing ("Print at 25%, sir — roughly X remaining")
# is the in-character variant the task spec called for. Rate-limiting via
# bambu_print_announcer keeps us from talking over those.
_MILESTONES = (25, 50, 75, 100)

# Error keywords we surface with the "I'm afraid..." phrasing. Scanned
# against the stringified AMS payload + print_error so we catch the
# common failure signatures regardless of firmware revision.
_LAYER_SHIFT_KEYWORDS = ("layer shift", "shifted", "shift detected")
_AMS_KEYWORDS         = ("ams", "spool", "filament")

# Per-print bookkeeping. Reset whenever bambu_monitor reports a new
# filename or a transition out of a terminal state into RUNNING.
_state_lock                  = threading.Lock()
_current_filename: list      = [None]
_announced_milestones: set   = set()
_announced_error_codes: set  = set()
_announced_layer_shift: list = [False]
_announced_ams_error: list   = [False]
_announced_failed: list      = [False]
_hook_registered: list       = [False]


# ── bambu_monitor bridge ─────────────────────────────────────────────
def _get_bambu_module():
    """Resolve the loaded bambu_monitor module, or None if absent."""
    return sys.modules.get("skill_bambu_monitor")


def _get_announcer_module():
    """Resolve bambu_print_announcer — used for its gated, rate-limited
    _proactive_announce() so our milestone speech coordinates with the
    other companion skills. None when the announcer hasn't loaded."""
    return sys.modules.get("skill_bambu_print_announcer")


def _format_minutes(minutes) -> str:
    """Prefer bambu_monitor's formatter so phrasing matches the rest of
    the H2D voice copy. Fall back to a local implementation when bambu
    isn't loaded yet (test harness, partial boot)."""
    bm = _get_bambu_module()
    if bm is not None and hasattr(bm, "_format_minutes"):
        try:
            return bm._format_minutes(minutes)
        except Exception:
            pass
    try:
        m = int(minutes)
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return ""
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    hours, rem = divmod(m, 60)
    if rem == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return (f"{hours} hour{'s' if hours != 1 else ''} and "
            f"{rem} minute{'s' if rem != 1 else ''}")


def _format_temp(t) -> str:
    bm = _get_bambu_module()
    if bm is not None and hasattr(bm, "_format_temp"):
        try:
            return bm._format_temp(t)
        except Exception:
            pass
    try:
        v = float(t)
    except (TypeError, ValueError):
        return ""
    if v < 1:
        return ""
    return f"{int(round(v))} degrees"


def _strip_filename(name: str) -> str:
    bm = _get_bambu_module()
    if bm is not None and hasattr(bm, "_strip_filename"):
        try:
            return bm._strip_filename(name)
        except Exception:
            pass
    if not name:
        return ""
    import re
    base = os.path.basename(name)
    base = re.sub(r"\.(3mf|gcode|gco)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_\-]+", " ", base).strip()
    return base[:60] if base else ""


def _read_state():
    """Snapshot bambu_monitor._state under its lock. Returns None when
    bambu_monitor isn't loaded so callers can degrade cleanly."""
    bm = _get_bambu_module()
    if bm is None:
        return None
    try:
        with bm._state_lock:
            return dict(bm._state)
    except Exception:
        return None


# ── speech routing ───────────────────────────────────────────────────
def _gated_announce(message: str) -> None:
    """Route milestone speech through bambu_print_announcer's gated
    _proactive_announce when available — that path coordinates focus
    mode + rate-limiting across the companion skills so we never stack
    a double announcement on top of bambu_monitor's own line. Falls
    back to the direct enqueue path when the announcer isn't loaded."""
    ann = _get_announcer_module()
    if ann is not None and hasattr(ann, "_proactive_announce"):
        try:
            ann._proactive_announce(message)
            return
        except Exception:
            pass
    _direct_enqueue(message)


def _direct_enqueue(message: str) -> None:
    """Bypass the rate-limit — used for error / FAILED / layer-shift
    callouts where suppressing the alert because a 50% milestone
    happened to fire 90s earlier would be the wrong trade-off.
    Prefer bobert_companion.proactive_announce when reachable so the
    main listen loop can voice it at the next turn boundary."""
    try:
        bc = importlib.import_module("bobert_companion")
        fn = getattr(bc, "proactive_announce", None)
        if callable(fn):
            fn(message, source="bambu_voice_companion")
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
    try:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        if not isinstance(data, list):
            data = []
        data.append({"ts": time.time(), "message": message})
        _atomic_write_json(_SPEECH_QUEUE, data)
    except Exception as e:
        print(f"  [bambu_voice] speech-queue write failed ({e}); alert: {message}")


# ── error pattern scan ───────────────────────────────────────────────
def _scan_for_layer_shift(err, ams) -> bool:
    """True when print_error or ams_status mentions a layer shift —
    the H2D's stepper-skip detector surfaces these as a distinct
    print_error keyword on recent firmware."""
    blob = ""
    if err and err not in (0, "0", None):
        blob += str(err).lower() + " "
    if ams is not None:
        try:
            blob += (json.dumps(ams).lower()
                     if not isinstance(ams, str) else ams.lower())
        except Exception:
            pass
    return any(kw in blob for kw in _LAYER_SHIFT_KEYWORDS)


def _scan_for_ams_issue(err, ams) -> bool:
    """True when ams_status carries a fault keyword. Distinct from
    bambu_print_announcer's runout/jam scan so we can phrase the line
    as a general "AMS appears to be unwell" rather than the specific
    runout copy that skill already owns."""
    if ams is None:
        return False
    try:
        blob = (json.dumps(ams).lower()
                if not isinstance(ams, str) else ams.lower())
    except Exception:
        return False
    if not any(kw in blob for kw in _AMS_KEYWORDS):
        return False
    # Only fire when there's an actual fault signature in the AMS
    # payload — the literal word "ams" appearing inside a healthy
    # status block shouldn't trigger the announcement.
    return any(kw in blob for kw in ("fault", "error", "jam", "runout",
                                     "stuck", "unwell"))


# ── state-change hook ────────────────────────────────────────────────
def _on_bambu_state_change(snapshot: dict, prev_gcode, gcode_state: str) -> None:
    """Hook callback registered with bambu_monitor.register_state_change_hook.

    Must complete quickly — bambu_monitor's on_message thread blocks on
    every hook before returning to MQTT processing. All speech routing
    is queue-based and non-blocking; the heavy lifting (TTS, focus mode
    checks) happens off-thread in the announcer / bobert_companion side.
    """
    try:
        with _state_lock:
            _process_snapshot(snapshot, gcode_state)
    except Exception as e:
        # Never let a bug in here suppress bambu_monitor's own callouts.
        print(f"  [bambu_voice] state-change hook crashed: {e}")


def _process_snapshot(snapshot: dict, gcode_state: str) -> None:
    """Walk a single bambu_monitor state snapshot and decide which
    announcements to queue. Caller holds _state_lock."""
    gcode_state = (gcode_state or "").upper()
    fname       = _strip_filename(snapshot.get("filename") or "")
    pct         = snapshot.get("mc_percent")
    remaining   = snapshot.get("mc_remaining")
    layer       = snapshot.get("layer_num")
    err         = snapshot.get("print_error")
    ams         = snapshot.get("ams_status")

    # New-print detection — keyed on filename change. We deliberately
    # don't try to detect a fresh start from gcode_state alone here
    # because bambu_monitor already does that and double-resetting our
    # bookkeeping mid-print would cause repeated announcements.
    if fname and fname != _current_filename[0]:
        _current_filename[0] = fname
        _announced_milestones.clear()
        _announced_error_codes.clear()
        _announced_layer_shift[0] = False
        _announced_ams_error[0]   = False
        _announced_failed[0]      = False

    # Coerce pct once.
    try:
        pct_f = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        pct_f = None

    # Milestones — gated through the rate-limited announcer path so we
    # don't double-speak on top of bambu_monitor's own 25/50/75 line.
    if gcode_state == "RUNNING" and pct_f is not None:
        for threshold in _MILESTONES:
            if pct_f >= threshold and threshold not in _announced_milestones:
                _announced_milestones.add(threshold)
                remaining_str = _format_minutes(remaining) if remaining else ""
                if remaining_str:
                    msg = (f"Print at {threshold}%, sir — roughly "
                           f"{remaining_str} remaining.")
                else:
                    msg = f"Print at {threshold}%, sir."
                _gated_announce(msg)
                break  # one milestone per snapshot

    # Layer shift — fires once per print, bypassing the rate limit
    # because mechanical failure isn't a milestone-priority event.
    if not _announced_layer_shift[0] and _scan_for_layer_shift(err, ams):
        _announced_layer_shift[0] = True
        layer_phrase = f" on layer {layer}" if layer else ""
        _direct_enqueue(
            f"Layer shift detected{layer_phrase}, sir — "
            f"I'm afraid we have a problem."
        )

    # AMS issue (non-runout fault). bambu_print_announcer already
    # handles runout specifically; this catches the more general AMS
    # unwellness so a jammed spool with no runout code still surfaces.
    if not _announced_ams_error[0] and _scan_for_ams_issue(err, ams):
        _announced_ams_error[0] = True
        _direct_enqueue(
            "I'm afraid the AMS appears to be unwell, sir — "
            "you may want to check the spools."
        )

    # FAILED — distinct from bambu_monitor's "Print has failed at layer
    # N" line. The character cue here is the apologetic framing.
    if gcode_state == "FAILED" and not _announced_failed[0]:
        _announced_failed[0] = True
        layer_phrase = f" at layer {layer}" if layer else ""
        _direct_enqueue(
            f"I'm afraid the print has failed{layer_phrase}, sir. "
            f"You'll want to take a look."
        )


# ── hook registration ────────────────────────────────────────────────
def _register_bambu_hook() -> bool:
    """Best-effort wire-up. Returns True when the hook is attached.
    Both failure modes (bambu missing, register call raising) leave the
    skill dormant — print_status still works, milestone announcements
    simply never fire."""
    bm = _get_bambu_module()
    if bm is None:
        return False
    fn = getattr(bm, "register_state_change_hook", None)
    if not callable(fn):
        return False
    try:
        fn(_on_bambu_state_change)
        _hook_registered[0] = True
        return True
    except Exception as e:
        print(f"  [bambu_voice] hook registration failed: {e}")
        return False


# ── action ───────────────────────────────────────────────────────────
def _build_print_status_line() -> str:
    """Compose a single-line JARVIS-style status reply, modelled on
    bambu_monitor.how_is_the_print but exposed under the more
    discoverable `print_status` action name."""
    state = _read_state()
    if state is None:
        return ("The H2D monitor isn't running, sir — I can't see the "
                "printer at the moment.")
    if state.get("last_update", 0.0) == 0.0:
        return ("I don't have a fresh status from the printer yet, sir. "
                "Either it isn't reachable or the monitor hasn't "
                "connected.")

    gcode_state = (state.get("gcode_state") or "").upper()
    layer       = state.get("layer_num")
    total       = state.get("total_layer")
    remaining   = state.get("mc_remaining")
    fname       = _strip_filename(state.get("filename") or "")
    nozzle      = state.get("nozzle_temper")
    bed         = state.get("bed_temper")

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


def register(actions: dict) -> None:
    def print_status(_: str = "") -> str:
        """Live H2D print snapshot: filename, layer/total, ETA, nozzle
        and bed temps. Named `print_status` for the obvious phrasing
        ('what's the print status?'); falls back gracefully when
        bambu_monitor isn't loaded."""
        return _build_print_status_line()

    actions["print_status"] = print_status

    # Wire into bambu_monitor's state-change dispatch. Failure is fine
    # — the skill still exposes `print_status` as a polled status query
    # even when no proactive announcements are wired up.
    _register_bambu_hook()
