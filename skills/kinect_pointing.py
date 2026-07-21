"""
kinect_pointing skill — "point-to-control" wiring for JARVIS via the Kinect v2.

THE FEATURE
===========
The owner points an arm at a real device (a desk lamp, a fan) and says
"turn that on" — and JARVIS controls the RIGHT smart-home device. It works in
two halves:

  1. CALIBRATION (once per device): the user points at the lamp and says
     "calibrate pointing for the desk lamp". JARVIS samples the pointing
     direction for ~1 s, averages it, and stores that direction bound to the
     real smart-home device named "desk lamp" (resolved against the discovered
     device catalog; if there's no exact match the spoken name is stored anyway
     and matched loosely at control time).

  2. CONTROL: the user points at the lamp and says "turn that on". JARVIS reads
     the live pointing direction, resolves it to the closest calibrated target
     within an angular cone (~18°), and fires the EXISTING smart-home on/off
     path for that target's bound device. No reimplementation of device
     control — it calls core.smart_home_router.smart_home_control("turn on the
     desk lamp"), exactly as if the user had named the device.

The pure geometry + the calibration store live in audio/kinect_pointing.py;
this skill is the live wiring around them — sensor contact, the staging gate,
the opt-in flag, the voice actions, and the "turn that on/off" natural-phrase
hook.

EVERYTHING is opt-in + safe (mirrors skills/kinect_gestures.py):
  • Gated by core.config.KINECT_POINT_CONTROL_ENABLED (default False), re-read
    each call so a Settings toggle takes effect with no restart.
  • A staging / test instance NEVER controls a real device (JARVIS_STAGING /
    bobert_companion._is_staging()).
  • All sensor contact is via audio/kinect_bridge (accessors never raise); a
    missing / disabled sensor degrades to an HONEST spoken no-op — it never
    pretends it pointed at something.
  • The calibration map is a SEPARATE gitignored json (data/kinect_pointing.json
    via JARVIS_POINTING_PATH) with an atomic write — device vectors never touch
    user_settings.json.

Voice actions:
  point_calibrate, <name>   — sample + store the direction the user is pointing,
                              bound to the named device. "calibrate pointing for
                              the desk lamp".
  list_point_targets / point_targets — read back what's calibrated. "what can I
                              point at".
  forget_point_target, <name> — drop one calibration.
  point_control, <on|off|toggle | utterance> — resolve the live point and fire
                              the smart-home action. Also the seam the natural
                              "turn that on" phrase routes through.
  point_status              — is point-control on + can I see you + how many
                              targets are calibrated.
  point_control_on / point_control_off — toggle KINECT_POINT_CONTROL_ENABLED
                              live, persisted via the same Settings writer
                              model_picker / kinect_gestures use.
"""
from __future__ import annotations

import os
import re
import sys
import time


# ─── tunables ────────────────────────────────────────────────────────────
# How long to sample the pointing direction during a calibration, and the poll
# cadence while sampling (mirrors the gesture poller's ~18 Hz).
CALIBRATE_SAMPLE_SECONDS = 1.0
CALIBRATE_POLL_HZ = 18.0
CALIBRATE_POLL_INTERVAL = 1.0 / CALIBRATE_POLL_HZ
# Hard cap on sampling wall-time so a wedged sensor read can't hang the voice
# loop (sample window + slack).
CALIBRATE_MAX_SECONDS = 2.5


# ─── module seams (lazy; never raise at import) ────────────────────────────
def _pointing_module():
    """The pure geometry + store module (audio.kinect_pointing). Imported
    lazily so a failure can't stop the skill registering its voice actions."""
    mod = sys.modules.get("audio.kinect_pointing")
    if mod is not None:
        return mod
    try:
        from audio import kinect_pointing as _kp
        return _kp
    except Exception:
        return None


def _bridge():
    """Live kinect_bridge module, or None. Prefer the instance the monolith
    already imported; fall back to a direct import (mirrors kinect_gestures)."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — point-control must NEVER drive a real
    device there. Matches the monolith's own gate plus the raw env var so the
    check holds even before the monolith is importable."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_is_staging", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    return False


def _point_control_enabled() -> bool:
    """The master gate: opt-in flag ON and not staging."""
    return _cfg_flag("KINECT_POINT_CONTROL_ENABLED") and not _is_staging()


# ─── calibration store accessor ────────────────────────────────────────────
def _store():
    """A fresh PointingStore bound to the configured (gitignored) json path, or
    None if the pure module is unavailable. Cheap to recreate — it reads the
    file per call so a concurrent calibrate is always seen."""
    kp = _pointing_module()
    if kp is None:
        return None
    try:
        return kp.PointingStore()
    except Exception:
        return None


# ─── sensor reads (all via the bridge; never raise) ────────────────────────
def _sensor_ready() -> tuple[bool, str]:
    """(True, "") when the Kinect is enabled AND available; else (False, why).
    `why` is a short honest reason for the spoken message."""
    kb = _bridge()
    if kb is None:
        return False, "the Kinect bridge isn't loaded"
    try:
        if not kb.get_enabled():
            return False, "the Kinect is switched off"
        ok, reason = kb.available()
        if not ok:
            return False, (reason or "the Kinect is unavailable")
    except Exception:
        return False, "the Kinect is unavailable"
    return True, ""


def _current_direction():
    """The live pointing direction (unit Vec3) from the NEAREST tracked body, or
    None when nothing is pointable. Never raises."""
    kb = _bridge()
    kp = _pointing_module()
    if kb is None or kp is None:
        return None
    try:
        bodies = kb.get_bodies()
    except Exception:
        return None
    body = _nearest_body(bodies)
    if body is None:
        return None
    try:
        ray = kp.arm_direction(body)
    except Exception:
        return None
    if ray is None:
        return None
    return ray[1]   # (origin, dir) → dir


def _nearest_body(bodies):
    """Closest tracked body from a get_bodies()-shaped list, or None. Reuses the
    pure module's helper when present (single source of truth); falls back to a
    local copy so this works even if that private name is ever renamed."""
    kp = _pointing_module()
    if kp is not None:
        fn = getattr(kp, "_nearest_body", None)
        if callable(fn):
            try:
                return fn(bodies)
            except Exception:
                pass
    if not bodies:
        return None
    try:
        cands = [b for b in bodies if isinstance(b, dict) and b.get("joints")]
    except TypeError:
        return None
    if not cands:
        return None

    def _key(b):
        d = b.get("distance_m")
        return d if isinstance(d, (int, float)) and d > 0 else float("inf")

    return min(cands, key=_key)


# ─── smart-home seam (call the EXISTING path — do not reimplement) ──────────
def _smart_home_control(utterance: str) -> str:
    """Fire JARVIS's existing smart-home dispatch for `utterance`. Prefers the
    live ACTIONS registry entry (honours any monolith wrapping); falls back to
    importing core.smart_home_router directly so point-control works even when
    invoked outside a full monolith boot. Returns the action's spoken result, or
    an honest error string."""
    # 1) Live registry (the same callable the LLM would invoke).
    bc = _bc()
    if bc is not None:
        actions = getattr(bc, "ACTIONS", None)
        if isinstance(actions, dict):
            for key in ("smart_home_control", "control_device",
                        "control_smart_home"):
                fn = actions.get(key)
                if callable(fn):
                    try:
                        return str(fn(utterance))
                    except Exception as e:
                        return f"the smart-home controller errored: {e}"
    # 2) Direct import of the canonical router.
    try:
        from core import smart_home_router as _shr
        return str(_shr.smart_home_control(utterance))
    except Exception as e:
        return f"the smart-home controller isn't reachable ({e})"


def _resolve_device_binding(target_name: str) -> str:
    """At calibration time, bind the spoken target name to a REAL smart-home
    device name when the catalog has a clear match (so control later addresses
    the exact device), else fall back to the spoken name itself (resolved
    loosely by smart_home_control at control time). Best-effort + read-only."""
    try:
        from core import smart_home_router as _shr
        catalog = _shr._ensure_catalog()
        if not catalog or not catalog.get("devices"):
            return target_name
        matches = _shr._resolve_devices(target_name, catalog)
        if matches:
            name = (matches[0].get("name") or "").strip()
            if name:
                return name
    except Exception:
        pass
    return target_name


# ─── speak seam (reuse the skill_utils hook, else the monolith) ────────────
def _speak(text: str) -> None:
    su = globals().get("skill_utils")
    if isinstance(su, dict):
        speaker = su.get("speak")
        if callable(speaker):
            try:
                speaker(text)
                return
            except Exception:
                pass
    bc = _bc()
    if bc is not None:
        try:
            fn = getattr(bc, "_speak", None) or getattr(bc, "speak", None)
            if callable(fn):
                fn(text)
        except Exception:
            pass


# ─── persistence (reuse the hardened Settings writer) ──────────────────────
def _persist_setting(key: str, value) -> bool:
    """Write {key: value} into the settings file WITHOUT clobbering the owner's
    other settings — the EXACT path model_picker / kinect_gestures use
    (settings_window.load_settings + save_settings, which honour
    JARVIS_SETTINGS_PATH so tests can't touch the real file). Best-effort."""
    try:
        from tools import settings_window as sw
    except Exception:
        return False
    try:
        current = sw.load_settings()
        if not isinstance(current, dict):
            current = {}
        current[key] = value
        sw.save_settings(current)
        return True
    except Exception:
        return False


def _set_enabled(on: bool) -> bool:
    """Flip KINECT_POINT_CONTROL_ENABLED live (core.config) and persist it."""
    try:
        import core.config as _cfg
        _cfg.KINECT_POINT_CONTROL_ENABLED = bool(on)
    except Exception:
        pass
    return _persist_setting("KINECT_POINT_CONTROL_ENABLED", bool(on))


# ─── calibration sampling ──────────────────────────────────────────────────
def _sample_direction(seconds: float = CALIBRATE_SAMPLE_SECONDS,
                      sleep_fn=time.sleep, now_fn=time.monotonic):
    """Gather pointing directions for ~`seconds`, then average them into one
    steady unit direction. Returns (direction|None, n_frames, n_pointing).

    n_frames is how many sensor ticks we read; n_pointing how many yielded a
    usable arm ray. A None direction means we either saw no pointing arm or the
    samples were too unsteady (per the pure averager's gate)."""
    kp = _pointing_module()
    if kp is None:
        return None, 0, 0
    dirs = []
    frames = 0
    deadline = now_fn() + min(seconds, CALIBRATE_MAX_SECONDS)
    while now_fn() < deadline:
        frames += 1
        d = _current_direction()
        if d is not None:
            dirs.append(d)
        sleep_fn(CALIBRATE_POLL_INTERVAL)
    try:
        avg = kp.average_direction(dirs)
    except Exception:
        avg = None
    return avg, frames, len(dirs)


# ─── actions ─────────────────────────────────────────────────────────────
def point_calibrate(arg: str = "") -> str:
    """Calibrate the pointing direction for a named device. The user points at
    the device and says e.g. 'calibrate pointing for the desk lamp' (arg = the
    device/target name). Samples for ~1 s, averages, binds to the real device,
    stores it. Honest on every failure — never claims to have learned a
    direction it didn't get."""
    name = _clean_target_name(arg)
    if not name:
        return ("Which device, sir? Try 'calibrate pointing for the desk "
                "lamp' while pointing at it.")
    if not _cfg_flag("KINECT_POINT_CONTROL_ENABLED"):
        return ("Point-to-control is off, sir — say 'turn on point control' "
                "first, then point at the device and calibrate it.")
    if _is_staging():
        return "Not while I'm in staging, sir."
    ready, why = _sensor_ready()
    if not ready:
        return (f"I can't calibrate that, sir — {why}. Enable the Kinect and "
                "try again while pointing at the device.")
    kp = _pointing_module()
    if kp is None:
        return "My pointing math module didn't load, sir — I can't calibrate."

    direction, frames, pointing = _sample_direction()
    if direction is None:
        if pointing == 0:
            return ("I couldn't see you pointing, sir — extend your arm toward "
                    "the device and hold it while I sample.")
        return ("Your arm was too unsteady for me to lock a direction, sir — "
                "hold the point still for a second and try again.")

    store = _store()
    if store is None:
        return "My calibration store didn't load, sir — I can't save that."
    device = _resolve_device_binding(name)
    ok = store.put(name, direction, device=device)
    if not ok:
        return (f"I read the direction to {name}, sir, but couldn't save the "
                "calibration to disk.")
    if device and device.lower() != name.lower():
        return (f"Got it — I'll remember the {name} is that way, sir, and "
                f"bind it to your '{device}' device.")
    return f"Got it — I'll remember the {name} is that way, sir."


def list_point_targets(_: str = "") -> str:
    """Read back the calibrated pointing targets. 'what can I point at'."""
    store = _store()
    if store is None:
        return "My calibration store didn't load, sir."
    targets = store.list_targets()
    if not targets:
        return ("Nothing's calibrated for pointing yet, sir — point at a device "
                "and say 'calibrate pointing for the desk lamp'.")
    names = []
    for t in targets:
        nm = t.get("name") or "?"
        dev = t.get("device")
        if dev and dev.lower() != nm.lower():
            names.append(f"{nm} (controls {dev})")
        else:
            names.append(nm)
    if len(names) == 1:
        return f"You can point at the {names[0]}, sir."
    return ("You can point at: " + ", ".join(names[:-1]) +
            f", and {names[-1]}, sir.")


def forget_point_target(arg: str = "") -> str:
    """Forget one calibrated pointing target by name."""
    name = _clean_target_name(arg)
    if not name:
        return "Which one should I forget, sir?"
    store = _store()
    if store is None:
        return "My calibration store didn't load, sir."
    if store.remove_target(name):
        return f"Forgotten — I'll no longer point-control the {name}, sir."
    return f"I had no pointing calibration for '{name}', sir."


def point_control(arg: str = "") -> str:
    """Resolve where the user is pointing and fire the smart-home on/off/toggle
    for the matched device. `arg` is the desired state: 'on' | 'off' | 'toggle'
    — OR a freeform utterance like 'turn that on' / 'that one off' from which
    the state is parsed. Honest when point-control is off, the sensor is
    absent, pointing isn't detected, or nothing calibrated matches.

    This is ALSO the seam the natural 'turn that on' phrase routes through (see
    resolve_pointing_command)."""
    if not _cfg_flag("KINECT_POINT_CONTROL_ENABLED"):
        return ("Point-to-control is off, sir — say 'turn on point control' to "
                "enable it, then point at a device and tell me on or off.")
    if _is_staging():
        return "Not while I'm in staging, sir."

    state = _parse_state(arg)
    if state is None:
        return ("On or off, sir? Point at the device and say 'turn that on' or "
                "'turn that off'.")

    ready, why = _sensor_ready()
    if not ready:
        return (f"I can't tell where you're pointing, sir — {why}.")

    direction = _current_direction()
    if direction is None:
        return ("I don't see you pointing at anything, sir — extend your arm "
                "toward the device and try again.")

    store = _store()
    if store is None:
        return "My calibration store didn't load, sir."
    target = store.resolve(direction)
    if target is None:
        if not store.list_targets():
            return ("Nothing's calibrated for pointing yet, sir — point at the "
                    "device and say 'calibrate pointing for the desk lamp' "
                    "first.")
        return ("You're not pointing at anything I've calibrated, sir — aim "
                "right at the device, or calibrate it first.")

    device = store.device_for(target) or target
    verb = "off" if state == "off" else "on"  # toggle handled below
    if state == "toggle":
        # No global device-state cache here; express the toggle to the
        # smart-home layer as a plain 'toggle <device>' so a brand skill that
        # supports it can flip, falling back to 'on' phrasing it understands.
        result = _smart_home_control(f"toggle {device}")
        if _looks_like_failure(result):
            result = _smart_home_control(f"turn on {device}")
        return _frame_result(target, device, "toggled", result)
    result = _smart_home_control(f"turn {verb} {device}")
    return _frame_result(target, device, verb, result)


def point_status(_: str = "") -> str:
    """Report whether point-control is on, whether a body is in view, and how
    many targets are calibrated. 'is point control on' / 'point status'."""
    enabled = _cfg_flag("KINECT_POINT_CONTROL_ENABLED")
    store = _store()
    n = len(store.list_targets()) if store is not None else 0
    cal = (f"{n} device{'s' if n != 1 else ''} calibrated" if n
           else "nothing calibrated yet")
    if not enabled:
        return (f"Point-to-control is off, sir — say 'turn on point control' to "
                f"enable it ({cal}).")
    ready, why = _sensor_ready()
    if not ready:
        return (f"Point-to-control is on, sir, but {why}, so I can't see where "
                f"you're pointing right now ({cal}).")
    pointing = _current_direction() is not None
    if not n:
        return ("Point-to-control is on, sir, but nothing's calibrated yet — "
                "point at a device and say 'calibrate pointing for the desk "
                "lamp'.")
    if pointing:
        return (f"Point-to-control is on and I can see you pointing, sir — "
                f"{cal}. Aim at one and say 'turn that on'.")
    return (f"Point-to-control is on, sir — {cal}. Point at one and say 'turn "
            "that on'.")


def point_control_on(_: str = "") -> str:
    """Turn point-to-control on (live + persisted)."""
    if _cfg_flag("KINECT_POINT_CONTROL_ENABLED"):
        already = "Point-to-control is already on, sir."
    else:
        already = None
    persisted = _set_enabled(True)
    ready, why = _sensor_ready()
    sensor_note = "" if ready else f" Note {why} — enable it so I can see you point."
    if already:
        return already + sensor_note
    msg = ("Point-to-control on, sir — point at a calibrated device and say "
           "'turn that on'.")
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg + sensor_note


def point_control_off(_: str = "") -> str:
    """Turn point-to-control off (live + persisted)."""
    if not _cfg_flag("KINECT_POINT_CONTROL_ENABLED"):
        return "Point-to-control is already off, sir."
    persisted = _set_enabled(False)
    msg = "Point-to-control off, sir."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


# ─── natural-phrase integration ("turn that on/off" via pointing) ──────────
# A pronoun device command ("turn that on", "that one off") should resolve via
# pointing BEFORE the smart-home router has to ask "which device?". The monolith
# (or anything routing a smart-home utterance) can call resolve_pointing_command
# first: if point-control is active AND the utterance is a bare pronoun command
# AND the user is pointing at a calibrated target, it executes via pointing and
# returns the spoken result. Otherwise it returns None and the caller's existing
# behaviour is completely unchanged (non-breaking, best-effort).

# Pronoun device phrases: "turn that on", "that one off", "turn this on",
# "switch that off", "that on", "this one on". Deliberately narrow — only
# matches when the TARGET is a pronoun (that/this/it/that one/this one), never a
# named device, so 'turn off the office light' never gets hijacked.
_PRONOUN_TARGET = r"(?:that|this|it)(?:\s+one)?"
# Lead-in fillers may STACK ("could you please ...") and the wake word may
# carry Whisper's comma ("JARVIS, turn that on") — same lead-filler rule as
# core.lead_fillers, in regex form (2026-07-21 audit, stale-duplicates).
_PRONOUN_COMMAND_RE = re.compile(
    r"^\s*(?:(?:could you|can you|please|jarvis|hey jarvis|would you),?\s+)*"
    r"(?:turn|switch|flip)?\s*"
    r"(?:(?P<on1>on|off)\s+" + _PRONOUN_TARGET + r"|"
    r"" + _PRONOUN_TARGET + r"\s+(?P<on2>on|off))"
    r"\s*(?:one)?\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def is_pronoun_device_command(utterance: str) -> bool:
    """True when `utterance` is an ambiguous pronoun on/off command ('turn that
    on', 'that one off') — the kind that, without pointing, would force a 'which
    device?' clarification. Pure + side-effect free."""
    if not utterance:
        return False
    return bool(_PRONOUN_COMMAND_RE.match(utterance.strip()))


def resolve_pointing_command(utterance: str):
    """Best-effort pointing resolution for an ambiguous pronoun command.

    Returns a spoken result string when ALL of:
      • point-control is enabled and not staging,
      • the utterance is a bare pronoun on/off command,
      • the Kinect can see the user pointing,
      • the point resolves to a calibrated target,
    and the smart-home action was fired. Returns None otherwise — signalling the
    caller to fall through to its existing (ask-which-device) behaviour with NO
    change. Never raises."""
    try:
        if not _point_control_enabled():
            return None
        if not is_pronoun_device_command(utterance):
            return None
        state = _parse_state(utterance)
        if state is None:
            return None
        ready, _why = _sensor_ready()
        if not ready:
            return None
        direction = _current_direction()
        if direction is None:
            return None
        store = _store()
        if store is None:
            return None
        target = store.resolve(direction)
        if target is None:
            return None
        # We have a live point AND a calibrated match → drive it.
        return point_control(state)
    except Exception:
        return None


# ─── small helpers ──────────────────────────────────────────────────────────
def _clean_target_name(arg: str) -> str:
    """Normalise a spoken target name from the action arg. Strips the common
    'pointing for' / 'for the' lead-ins and trailing 'lamp/light' is KEPT (it's
    part of the device name). Returns '' when nothing usable remains."""
    s = (arg or "").strip()
    if not s:
        return ""
    low = s.lower()
    # Strip a leading "pointing for"/"for"/"the" the user (or the LLM) may have
    # left in the arg: 'pointing for the desk lamp' → 'desk lamp'.
    for lead in ("pointing for the ", "pointing for ", "for the ", "for ",
                 "the ", "a ", "an "):
        if low.startswith(lead):
            s = s[len(lead):]
            low = s.lower()
            break
    return s.strip()


def _parse_state(arg: str):
    """Extract the desired state from the arg: 'on' | 'off' | 'toggle' | None.
    Accepts bare states and pronoun phrasings ('turn that on', 'that one off',
    'toggle')."""
    s = (arg or "").strip().lower()
    if not s:
        return None
    if s in ("on", "off", "toggle"):
        return s
    if re.search(r"\btoggle\b", s):
        return "toggle"
    # Prefer an explicit on/off token anywhere in a short pronoun phrase.
    if re.search(r"\boff\b", s):
        return "off"
    if re.search(r"\bon\b", s):
        return "on"
    return None


def _looks_like_failure(result: str) -> bool:
    """Heuristic: did a smart_home_control reply read as a failure? Used so a
    'toggle' that a brand skill doesn't support falls back to 'on'."""
    if not result:
        return True
    low = result.lower()
    return any(p in low for p in (
        "didn't work", "i don't see anything", "couldn't parse",
        "no smart-home catalog", "isn't reachable", "errored",
        "i need something to do",
    ))


def _frame_result(target: str, device: str, verb: str, result: str) -> str:
    """Wrap the smart-home action's reply so the user hears WHICH pointed-at
    device acted. If the smart-home layer clearly failed, surface that honestly
    rather than claiming success."""
    if _looks_like_failure(result):
        # Pass the underlying reason through — it's already user-facing.
        return (f"I see you pointing at the {target}, sir, but controlling it "
                f"didn't go through: {result}")
    # Success: the underlying summary already names the device + state; lead with
    # the pointing acknowledgement so it's clear the POINT drove it.
    return result


# ─── registration ────────────────────────────────────────────────────────
def register(actions):
    actions["point_calibrate"]      = point_calibrate
    actions["calibrate_pointing"]   = point_calibrate
    actions["list_point_targets"]   = list_point_targets
    actions["point_targets"]        = list_point_targets
    actions["forget_point_target"]  = forget_point_target
    actions["point_control"]        = point_control
    actions["point_at"]             = point_control
    actions["point_status"]         = point_status
    actions["point_control_on"]     = point_control_on
    actions["point_control_off"]    = point_control_off
    print("  [point-control] point-to-control actions registered "
          "(opt-in via KINECT_POINT_CONTROL_ENABLED, off by default)")
