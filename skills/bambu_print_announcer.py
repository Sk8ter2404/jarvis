"""
Bambu print announcer — extra milestones + pause/resume controls +
the gated `proactive_print_announcer` layer.

skills/bambu_monitor.py already handles the MQTT connection and announces the
25 / 50 / 75 % milestones, FINISH, FAILED, layer-1 adhesion, in-flight errors,
and the bed-cool follow-up. This skill rides on top of that:

  • Adds the 10 % and 95 % milestone callouts (with layer + ETA when known).
  • Adds a separate first-layer adhesion check phrased as the spec example
    ('Print layer N of M — first layer adhesion appears nominal') so the
    callout fires at a few early-print layers, not just layer 2.
  • The `proactive_print_announcer` layer adds JARVIS-voice callouts for
    filament runout, AMS faults, and a celebratory 'Your part is ready, sir.'
    on completion — gated by `dnd_focus_mode.is_focus_mode_active()` and
    rate-limited to one announcement per ANNOUNCER_RATE_LIMIT_SECONDS (10 min
    by default) so a chatty print doesn't drown out everything else.
  • Registers `pause_print` and `resume_print` actions that publish printer
    commands over the existing MQTT client.
  • Registers `proactive_announcer_status` so the user can ask
    'announcer status' and find out whether focus mode or rate-limit is
    currently suppressing callouts.

It reads bambu_monitor's shared `_state` dict (under its lock) so we never
double-connect or duplicate the announcements bambu_monitor already owns.
The original spec mentioned 'Bambu Studio hotkeys', but Bambu Studio doesn't
expose stable pause/resume keybindings and the printer responds directly to
MQTT regardless of whether Studio is running — so pause/resume go to the
printer.
"""
import importlib
import json
import logging
import os
import sys
import threading
import time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# How often to peek at bambu_monitor._state. Milestones only need eventual
# consistency and bambu_monitor itself fires on every MQTT push, so we can
# afford to be lazy here.
POLL_INTERVAL_SECONDS = 20.0
# Give bambu_monitor's MQTT client time to connect and receive the first push
# before we start checking milestones; otherwise the very first poll might
# read a half-populated _state on cold start.
INITIAL_DELAY_SECONDS = 45.0

# Milestones owned by THIS skill. 25/50/75 stay with bambu_monitor to avoid
# double-announcing; we only fill the 10 % and 95 % gaps.
EXTRA_MILESTONES = [
    (10, "Sir, the print is 10% complete.{layer}{eta}"),
    (95, "The print is 95% complete, sir.{layer}{eta} We're nearly there."),
]

# Optional layer-based callouts during the failure-prone early layers. Each
# entry is the layer number at which to remark on adhesion, phrased like the
# task spec example. Fires only once per print and only when total_layer is
# known so the 'N of M' phrasing makes sense.
EARLY_LAYER_CHECKPOINTS = (5, 10)

# proactive_print_announcer gating: one callout per 10 minutes, suppressed
# entirely when focus mode is active. Critical alerts (FAILED, print_error
# codes) still ride through bambu_monitor's separate _enqueue_speech path,
# so this rate-limit can't silence a hard failure.
ANNOUNCER_RATE_LIMIT_SECONDS = 600.0

# AMS / print-error substrings we treat as a filament runout. Bambu's error
# codes are typed in the firmware as 4-group hex; the runout family begins
# with these prefixes (0300_03xx, 0500_03xx). We also fall back to a keyword
# scan of ams_status so an arbitrary firmware revision still trips the
# announcement.
_RUNOUT_ERROR_PREFIXES = ("0300_03", "0500_03")
_RUNOUT_AMS_KEYWORDS   = ("runout", "ran out", "empty", "no filament")

# AMS hard-fault keywords — distinct from runout so the announcement copy
# can be specific ('AMS reporting a fault' vs 'Filament has run out').
_AMS_FAULT_KEYWORDS = ("jam", "fault", "error", "stuck")

_announced_pct: set = set()
_announced_layers: set = set()
_current_filename = [None]
_armed_for_new_print = [False]
# proactive_print_announcer per-print state
_saw_running_this_print = [False]
_announced_runout       = [False]
_announced_ams_fault    = [False]
_announced_completion   = [False]
# Per-module rate-limit timestamp + lock
_rate_limit_lock     = threading.Lock()
_last_announcement_at = [0.0]
_last_suppressed_reason = [""]  # for proactive_announcer_status

_stop_evt = threading.Event()
_thread = [None]


def _get_bambu_module():
    return sys.modules.get("skill_bambu_monitor")


def _read_state():
    bm = _get_bambu_module()
    if bm is None:
        return None
    try:
        with bm._state_lock:
            return dict(bm._state)
    except Exception:
        return None


def _format_minutes(minutes) -> str:
    bm = _get_bambu_module()
    if bm is not None:
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


def _is_focus_active() -> bool:
    """True if skills/dnd_focus_mode.py is loaded and currently engaged.
    Falls back to False if the skill never loaded (e.g. the user disabled
    it) so absence of the helper never silences our callouts."""
    mod = sys.modules.get("skill_dnd_focus_mode")
    if mod is None:
        try:
            mod = importlib.import_module("skills.dnd_focus_mode")
        except Exception:
            return False
    fn = getattr(mod, "is_focus_mode_active", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _proactive_announce(message: str) -> bool:
    """Gated, rate-limited speech for the proactive_print_announcer layer.

    Drops the announcement if focus mode is active or another announcement
    fired within the last ANNOUNCER_RATE_LIMIT_SECONDS. Returns True when
    the message was queued for speech, False when suppressed. Callers that
    track milestone bookkeeping should treat suppressed announcements as
    'consumed' rather than retried — a 10-minute throttle isn't a defer.
    """
    if _is_focus_active():
        _last_suppressed_reason[0] = "focus mode"
        print(f"  [bambu_announcer] suppressed (focus mode): "
              f"{message[:120]}")
        return False
    now = time.time()
    with _rate_limit_lock:
        wait = ANNOUNCER_RATE_LIMIT_SECONDS - (now - _last_announcement_at[0])
        if wait > 0:
            _last_suppressed_reason[0] = (
                f"rate-limited ({int(wait)}s remaining)"
            )
            print(f"  [bambu_announcer] rate-limited "
                  f"(wait {int(wait)}s): {message[:120]}")
            return False
        _last_announcement_at[0] = now
    _last_suppressed_reason[0] = ""
    _enqueue_speech(message)
    return True


def _enqueue_speech(message: str) -> None:
    bm = _get_bambu_module()
    if bm is not None and hasattr(bm, "_enqueue_speech"):
        try:
            bm._enqueue_speech(message)
            return
        except Exception:
            pass
    # Last resort: write straight to the speech queue.
    queue_path = os.path.join(_PROJECT_DIR, "pending_speech.json")
    try:
        data = []
        if os.path.exists(queue_path):
            with open(queue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.append({"ts": time.time(), "message": message})
        # Atomic write (mkstemp+fsync+os.replace) so a concurrent reader never
        # catches a half-written queue and deletes it as corrupt — matches the
        # canonical proactive_announce / bambu_monitor writers.
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(queue_path, data)
    except Exception as e:
        print(f"  [bambu_announcer] speech-queue write failed ({e}); "
              f"alert: {message}")


def _read_config():
    """Pull config out of bambu_monitor so we use the same source of truth."""
    bm = _get_bambu_module()
    if bm is None or not hasattr(bm, "_read_config"):
        return "", "", ""
    try:
        return bm._read_config()
    except Exception:
        return "", "", ""


def _send_print_command(command: str) -> tuple:
    """Publish a print command (pause/resume/stop) to the printer over the
    MQTT client bambu_monitor already maintains.

    Returns (ok, error_message).
    """
    bm = _get_bambu_module()
    if bm is None:
        return False, "bambu monitor not loaded"
    client = None
    holder = getattr(bm, "_mqtt_client", None)
    if isinstance(holder, list) and holder:
        client = holder[0]
    if client is None:
        return False, "no MQTT client connected"
    # The paho client object exists immediately after connect_async(), so its
    # presence doesn't mean the session is up. If we publish before on_connect
    # fires, paho returns MQTT_ERR_NO_CONN WITHOUT raising and the command is
    # silently dropped — so gate on bambu_monitor's connected flag first.
    connected = getattr(bm, "_mqtt_connected_ok", None)
    if isinstance(connected, list):
        connected = connected[0] if connected else False
    if not connected:
        return False, "printer not connected"
    _ip, _access, serial = _read_config()
    if not serial:
        return False, "BAMBU_SERIAL not configured"
    try:
        topic = f"device/{serial}/request"
        payload = json.dumps({
            "print": {
                "sequence_id": "0",
                "command": command,
            }
        })
        info = client.publish(topic, payload)
        # publish() returns an MQTTMessageInfo whose .rc is non-zero (e.g.
        # MQTT_ERR_NO_CONN) when the message couldn't be queued for delivery.
        # Don't claim success on a dropped command.
        rc = getattr(info, "rc", 0)
        if rc != 0:
            return False, "printer not connected"
        return True, ""
    except Exception as e:
        return False, str(e)


def _poll_loop() -> None:  # pragma: no cover - daemon poll loop; blocks on _stop_evt.wait(POLL_INTERVAL_SECONDS) between live peeks at bambu_monitor._state. Its one work step, _check_milestones(), is unit-tested directly.
    if _stop_evt.wait(INITIAL_DELAY_SECONDS):
        return
    while not _stop_evt.is_set():
        try:
            try:
                _check_milestones()
            except Exception as e:
                print(f"  [bambu_announcer] poll loop error: {e}")
            if _stop_evt.wait(POLL_INTERVAL_SECONDS):
                return
        except Exception:
            logging.exception("[bambu_announcer] _poll_loop iteration crashed")
            if _stop_evt.wait(POLL_INTERVAL_SECONDS):
                return


def _check_milestones() -> None:
    state = _read_state()
    if state is None:
        return
    if state.get("last_update", 0.0) == 0.0:
        return

    gcode_state = (state.get("gcode_state") or "").upper()
    fname = state.get("filename") or ""
    pct = state.get("mc_percent")
    layer = state.get("layer_num")
    total = state.get("total_layer")
    remaining = state.get("mc_remaining")
    err = state.get("print_error")
    ams = state.get("ams_status")

    # Reset milestone bookkeeping when the print file changes — same trigger
    # bambu_monitor uses. We also reset when gcode_state moves out of a
    # terminal state into RUNNING, so a same-filename re-run isn't silenced.
    if fname and fname != _current_filename[0]:
        _current_filename[0] = fname
        _announced_pct.clear()
        _announced_layers.clear()
        _armed_for_new_print[0] = True
        _saw_running_this_print[0] = False
        _announced_runout[0] = False
        _announced_ams_fault[0] = False
        _announced_completion[0] = False

    # If JARVIS comes up mid-print, suppress past milestones so we don't blurt
    # 10 % the instant we discover an 80 %-done print.
    if not _armed_for_new_print[0] and gcode_state == "RUNNING":
        _armed_for_new_print[0] = True
        try:
            cur_pct = float(pct) if pct is not None else None
        except (TypeError, ValueError):
            cur_pct = None
        if cur_pct is not None:
            for threshold, _t in EXTRA_MILESTONES:
                if cur_pct >= threshold:
                    _announced_pct.add(threshold)
        try:
            cur_layer = int(layer) if layer is not None else None
        except (TypeError, ValueError):
            cur_layer = None
        if cur_layer is not None:
            for lp in EARLY_LAYER_CHECKPOINTS:
                if cur_layer >= lp:
                    _announced_layers.add(lp)
        # Mid-flight discovery: don't fire a celebratory "Your part is ready"
        # if we never actually saw this print run.
        _saw_running_this_print[0] = False
        return  # skip the actual announcement pass on the priming poll

    if gcode_state == "RUNNING":
        _saw_running_this_print[0] = True

    # ── proactive_print_announcer extras (filament runout, AMS faults,
    #    celebratory completion) — fire regardless of RUNNING/PAUSE/FINISH
    #    state so they aren't silenced if the printer pauses on runout.
    _check_runout_and_ams(err, ams, gcode_state)
    _check_celebratory_completion(gcode_state)

    if gcode_state != "RUNNING":
        return

    try:
        pct_f = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        pct_f = None

    layer_phrase = ""
    if layer and total:
        layer_phrase = f" Print layer {layer} of {total}."

    remaining_str = _format_minutes(remaining) if remaining else ""
    eta_phrase = f" Estimated {remaining_str} remaining." if remaining_str else ""

    # Percentage milestones — at most one per poll so announcements don't stack.
    # Marked as 'announced' even when _proactive_announce suppresses (focus
    # mode / rate limit), because a 10-minute throttle isn't a deferral.
    if pct_f is not None:
        for threshold, template in EXTRA_MILESTONES:
            if pct_f >= threshold and threshold not in _announced_pct:
                _announced_pct.add(threshold)
                _proactive_announce(template.format(layer=layer_phrase,
                                                    eta=eta_phrase))
                break

    # Early-layer adhesion checkpoint (only when total_layer is known so the
    # 'N of M' phrasing is meaningful).
    if layer and total:
        try:
            layer_i = int(layer)
        except (TypeError, ValueError):
            layer_i = None
        if layer_i is not None:
            for lp in EARLY_LAYER_CHECKPOINTS:
                if layer_i >= lp and lp not in _announced_layers:
                    _announced_layers.add(lp)
                    _proactive_announce(
                        f"Print layer {layer} of {total} — "
                        f"first layer adhesion appears nominal, sir."
                    )
                    break


def _check_runout_and_ams(err, ams, gcode_state: str) -> None:
    """Inspect print_error + ams_status and queue runout / AMS-fault
    announcements. Each fires at most once per print (cleared on new_print).
    """
    if _announced_runout[0] and _announced_ams_fault[0]:
        return

    # Stringify both candidates once for keyword scanning.
    err_str = ""
    if err and err not in (0, "0"):
        err_str = str(err).lower()
    ams_str = ""
    if ams is not None:
        try:
            ams_str = (json.dumps(ams).lower()
                       if not isinstance(ams, str) else ams.lower())
        except Exception:
            ams_str = ""

    # Filament runout — check error code prefixes first (firmware-specific
    # but most reliable), then fall back to keyword scan on AMS payload.
    if not _announced_runout[0]:
        runout = False
        if err_str:
            for prefix in _RUNOUT_ERROR_PREFIXES:
                if prefix in err_str:
                    runout = True
                    break
        if not runout and ams_str:
            for kw in _RUNOUT_AMS_KEYWORDS:
                if kw in ams_str:
                    runout = True
                    break
        if runout:
            _announced_runout[0] = True
            _proactive_announce(
                "Filament has run out, sir — the H2D is waiting on you."
            )
            # Treat a runout as also covering the AMS-fault announcement
            # so we don't double-speak about the same incident.
            _announced_ams_fault[0] = True
            return

    # AMS fault (non-runout) — only fire if ams_status carries a fault
    # keyword and we didn't already report runout. We deliberately ignore
    # `err_str` here because generic print errors are bambu_monitor's
    # territory; this skill only adds context when AMS is specifically
    # surfaced.
    if not _announced_ams_fault[0] and ams_str:
        for kw in _AMS_FAULT_KEYWORDS:
            if kw in ams_str:
                _announced_ams_fault[0] = True
                _proactive_announce(
                    "I'm afraid the AMS is reporting a fault, sir — "
                    "the spools want a look."
                )
                break


def _check_celebratory_completion(gcode_state: str) -> None:
    """Once per print, when we transition into FINISH after having
    observed the print actually running, emit a celebratory callout."""
    if _announced_completion[0]:
        return
    if gcode_state != "FINISH":
        return
    if not _saw_running_this_print[0]:
        # Don't celebrate a finish JARVIS only discovered post-hoc.
        return
    _announced_completion[0] = True
    _proactive_announce("Your part is ready, sir.")


def _start_poller():
    if _thread[0] is not None and _thread[0].is_alive():
        return
    _stop_evt.clear()
    t = threading.Thread(target=_poll_loop, daemon=True,
                         name="bambu_announcer")
    t.start()
    _thread[0] = t


def register(actions):
    def pause_print(_: str = "") -> str:
        state = _read_state() or {}
        gcode_state = (state.get("gcode_state") or "").upper()
        if gcode_state == "PAUSE":
            return "The print is already paused, sir."
        if gcode_state not in ("RUNNING",):
            return "No active print to pause, sir."
        ok, err = _send_print_command("pause")
        if not ok:
            return f"I couldn't reach the printer to pause it, sir — {err}."
        return "Pausing the print, sir."

    def resume_print(_: str = "") -> str:
        state = _read_state() or {}
        gcode_state = (state.get("gcode_state") or "").upper()
        if gcode_state == "RUNNING":
            return "The print is already running, sir."
        if gcode_state != "PAUSE":
            return "No paused print to resume, sir."
        ok, err = _send_print_command("resume")
        if not ok:
            return f"I couldn't reach the printer to resume it, sir — {err}."
        return "Resuming the print, sir."

    def proactive_announcer_status(_: str = "") -> str:
        """Report whether the proactive_print_announcer layer is currently
        suppressing callouts, and why. Useful when the user expected an
        announcement and didn't get one."""
        if _is_focus_active():
            return ("Proactive print announcer is suppressed, sir — "
                    "focus mode is active.")
        with _rate_limit_lock:
            wait = ANNOUNCER_RATE_LIMIT_SECONDS - (
                time.time() - _last_announcement_at[0]
            )
        if wait > 0:
            mins = int(wait // 60)
            secs = int(wait % 60)
            tail = (f"{mins} minute{'s' if mins != 1 else ''}"
                    if mins else f"{secs} seconds")
            return (f"Announcer is rate-limited, sir — next callout "
                    f"available in about {tail}.")
        return "Proactive print announcer is armed and ready, sir."

    actions["pause_print"] = pause_print
    actions["resume_print"] = resume_print
    actions["proactive_announcer_status"] = proactive_announcer_status

    _start_poller()
