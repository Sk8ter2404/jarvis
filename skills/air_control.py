"""
air_control skill — movie-style Kinect v2 spatial hand-mouse ("AIR CONTROL").

THE FEATURE  (REACH-TO-ENGAGE model; distinct from skills/kinect_air_mouse)
===========================================================================
REACH a hand OUT toward the sensor (arm extended past the shoulder) AND above
the waist to take the cursor — it then follows the hand across the WHOLE
virtual desktop (every monitor). CLOSE the hand into a FIST to GRAB whatever's
under the cursor and DRAG it; OPEN the hand to drop it. A QUICK close→open with
little travel is a CLICK. A LASSO (two-finger pointing) hand SCROLLS with
vertical motion. LOWER or RETRACT the hand and the cursor is released entirely
(any held button let go — the dead-man).

This module is ONLY the live wiring; ALL of the decision logic (engagement +
hysteresis, body-relative box → desktop mapping, EMA smoothing, the
grab/click/drag/scroll state machine) lives in the PURE engine
core/air_control.py — stdlib-only, sensor-free, unit-tested in
tests/test_air_control.py. Per tick this loop simply does:

    bodies = audio.kinect_bridge.get_bodies()      # the bridge is the contract
    op     = engine.update(bodies, _virtual_bounds())
    _apply_op(op)                                   # pyautogui move/down/up/...

SAFETY MODEL (mirrors the sibling Kinect skills, and then some)
===============================================================
  • core.config.AIR_CONTROL_ENABLED defaults to False. The skill ALWAYS loads
    (so the voice actions exist), but the control LOOP only auto-starts at load
    when the knob is True. When it's False, saying "air control on" STILL starts
    the loop — the explicit voice command is the owner's consent; the knob only
    guards against the mouse being driven UNINVITED at boot by a Kinect glitch.
  • FAILSAFE: ANY exception inside the loop releases both held-button
    possibilities (pyautogui.mouseUp) and STOPS the loop with a log line —
    a crash can never strand a grabbed window or leave a runaway cursor.
  • Stopping (voice "air control off", or the failsafe) always force-releases
    via engine.release() + mouseUp before the thread exits.
  • A staging / test instance NEVER moves the mouse (JARVIS_STAGING gate,
    same as kinect_air_mouse) — the loop refuses to start there.
  • No Kinect / bridge unavailable → the actions reply gracefully
    ("Kinect isn't available, sir.") and nothing runs.

Voice actions (all three return ONE finished sentence, so they're listed in
bobert_companion.SPEAK_RESULT_VERBATIM_ACTIONS — spoken verbatim, never
re-summarised):
  air_control_on / air_control_off / air_control_status
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

# The pure engine. Import defensively so a (should-never-happen) engine load
# failure disables the feature without taking down the skill loader — the same
# pattern gpu_usage uses for core.gpu_usage.
try:
    from core.air_control import AirControlEngine, OP_CLICK, OP_DOWN, OP_IDLE, \
        OP_MOVE, OP_SCROLL, OP_UP
    _HAS_ENGINE = True
except Exception as _exc:   # pragma: no cover - core.air_control is in-tree
    AirControlEngine = None     # type: ignore[assignment]
    _HAS_ENGINE = False
    print(f"  [air-control] core.air_control unavailable ({_exc}); "
          f"feature disabled")


# ─── tunables ────────────────────────────────────────────────────────────
AIR_CONTROL_POLL_HZ = 30.0                    # engine tick rate (~Kinect body rate)
AIR_CONTROL_POLL_INTERVAL = 1.0 / AIR_CONTROL_POLL_HZ
_THREAD_NAME = "air-control-skill"

# Win32 GetSystemMetrics indices for the VIRTUAL desktop (all monitors) — the
# ctypes fallback when the monolith's _virtual_screen_bounds isn't reachable.
# Same constants skills/kinect_air_mouse.py documents: x/y are NEGATIVE when a
# monitor sits left of / above the primary; pyautogui/SetCursorPos accept them.
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


# ─── loop state (module-level so on/off/status all see the same loop) ────
_stop_event = threading.Event()
_loop_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()      # serialise start/stop from voice actions
# One engine per loop run; kept module-level so air_control_status can report
# engagement even mid-run. Recreated on every start so state is always clean.
_engine: Optional["AirControlEngine"] = None
_last_stop_reason: str = ""          # why the loop last stopped (status/debug)


# ─── live-environment helpers (mirror skills/kinect_air_mouse.py) ────────
def _bridge():
    """Live kinect_bridge module, or None. Prefer the instance the monolith
    already imported; fall back to a direct import (mirrors kinect_gestures /
    kinect_air_mouse)."""
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


# ─── AUTO-YIELD bridge (skills/_air_mouse_yield, shared with air-mouse) ────
# "Touch the real mouse and air control instantly lets go." Same low-level
# real-input watcher kinect_air_mouse uses; every accessor degrades to a safe
# default (no yield suppression) when the helper can't load. NEVER raises.
AIR_CONTROL_YIELD_WINDOW_SEC = 1.0   # stay suppressed this long after real input


def _yield_mod():
    try:
        from skills import _air_mouse_yield as _y
        return _y
    except Exception:
        try:
            import _air_mouse_yield as _y   # isolated-skill import fallback
            return _y
        except Exception:
            return None


def _install_yield_watcher() -> None:
    """Lazily install the real-input hook (idempotent, safe every start)."""
    y = _yield_mod()
    if y is None:
        return
    try:
        y.install()
    except Exception:
        pass


def _real_input_recent() -> bool:
    """True when REAL (non-injected) hardware input happened within the yield
    window — air control must release and stay suppressed. False when the
    watcher is unavailable."""
    y = _yield_mod()
    if y is None:
        return False
    try:
        return bool(y.real_input_recent(AIR_CONTROL_YIELD_WINDOW_SEC))
    except Exception:
        return False


def _mark_self_action() -> None:
    """Tell the watcher our own injected mouse ops aren't the owner's input
    (so its polling fallback can't mistake us for a real hand on the mouse)."""
    y = _yield_mod()
    if y is None:
        return
    try:
        y.mark_self_action()
    except Exception:
        pass


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — air control must NEVER move the real
    cursor there. Same gate kinect_air_mouse uses (env var first so it holds
    even before the monolith is importable)."""
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


def _sensor_ready() -> tuple:
    """(True, "") when the Kinect is enabled AND available; else (False, why).
    Same shape as kinect_air_mouse._sensor_ready so the spoken phrasing can
    reuse the bridge's honest reason strings."""
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


def _virtual_bounds() -> tuple:
    """(x, y, w, h) of the whole virtual desktop spanning ALL monitors.
    Prefer the monolith's _virtual_screen_bounds (single source of truth for
    the MONITORS layout / DPI handling), then ctypes user32 directly, then a
    safe primary-ish default. NEVER raises."""
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_virtual_screen_bounds", None)
        if callable(fn):
            try:
                vx, vy, vw, vh = fn()
                if vw > 0 and vh > 0:
                    return int(vx), int(vy), int(vw), int(vh)
            except Exception:
                pass
    try:
        import ctypes
        gsm = ctypes.windll.user32.GetSystemMetrics
        vx = int(gsm(_SM_XVIRTUALSCREEN))
        vy = int(gsm(_SM_YVIRTUALSCREEN))
        vw = int(gsm(_SM_CXVIRTUALSCREEN))
        vh = int(gsm(_SM_CYVIRTUALSCREEN))
        if vw > 0 and vh > 0:
            return vx, vy, vw, vh
    except Exception:
        pass
    return 0, 0, 2560, 1440


# ─── mouse actuation (pyautogui; imported lazily so tests can mock it) ───
def _pyautogui():
    """The pyautogui module with its corner-failsafe DISARMED, or None when it
    can't import (headless CI). Disarming FAILSAFE is deliberate and safe here:
    the engine legitimately maps a full hand sweep to the desktop CORNERS
    (clamp), which pyautogui's default failsafe would misread as a panic-abort
    and raise mid-loop — our own dead-man (drop the hand) + the loop failsafe
    are the real safety net, same rationale as kinect_air_mouse's fallback."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        return pyautogui
    except Exception:
        return None


def _release_mouse() -> None:
    """FAILSAFE release: let go of any possibly-held button. Called on loop
    stop, on any loop exception, and by air_control_off — releasing an
    un-pressed button is a harmless no-op, so this is always safe to call.
    NEVER raises."""
    pg = _pyautogui()
    if pg is None:
        return
    try:
        pg.mouseUp(button="left")
    except Exception:
        pass


def _apply_op(op) -> None:
    """Execute ONE engine AirOp with pyautogui. May raise — the loop's
    failsafe wrapper catches, releases, and stops (see _loop)."""
    if op is None or op.kind == OP_IDLE:
        return
    pg = _pyautogui()
    if pg is None:
        return
    if op.kind == OP_MOVE and op.x is not None:
        # _pause=False isn't used: pyautogui.PAUSE default (0.1 s) would halve
        # our 30 Hz — so zero the module pause once per call instead.
        pg.PAUSE = 0
        pg.moveTo(int(op.x), int(op.y))
    elif op.kind == OP_DOWN and op.x is not None:
        pg.PAUSE = 0
        pg.moveTo(int(op.x), int(op.y))
        pg.mouseDown(button="left")
    elif op.kind == OP_UP:
        pg.mouseUp(button="left")
    elif op.kind == OP_CLICK and op.x is not None:
        pg.PAUSE = 0
        pg.click(int(op.x), int(op.y), button="left")
    elif op.kind == OP_SCROLL and op.scroll_amount:
        pg.scroll(int(op.scroll_amount))


# ─── the control loop ─────────────────────────────────────────────────────
def _loop() -> None:
    """The ~30 Hz daemon loop: poll bodies → engine.update → apply the op.

    FAILSAFE CONTRACT: any exception ANYWHERE in the tick releases held
    buttons and STOPS the loop with a log line (never a silent zombie driving
    the mouse). Normal exit (stop_event set by air_control_off) also releases,
    via engine.release() + _release_mouse()."""
    global _last_stop_reason
    engine = _engine
    try:
        _install_yield_watcher()
        while not _stop_event.is_set():
            # AUTO-YIELD: the owner touching the real mouse/keyboard wins
            # instantly — release any held drag, discard this tick's op, and
            # stay hands-off until the yield window passes. The engine still
            # runs (so its state stays coherent); we just don't apply ops.
            if _real_input_recent():
                try:
                    if engine is not None and engine.release() is not None:
                        _release_mouse()
                except Exception:
                    _release_mouse()
                time.sleep(AIR_CONTROL_POLL_INTERVAL)
                continue
            kb = _bridge()
            bodies = kb.get_bodies() if kb is not None else []
            op = engine.update(bodies, _virtual_bounds()) if engine else None
            _apply_op(op)
            if op is not None and op.kind != OP_IDLE:
                _mark_self_action()
            time.sleep(AIR_CONTROL_POLL_INTERVAL)
        _last_stop_reason = "stopped by request"
    except Exception as e:
        # FAILSAFE: never leave a button held or keep driving the mouse after
        # an error. Log loudly (this is the line the task's safety spec names).
        _last_stop_reason = f"failsafe stop: {e}"
        print(f"  [air-control] LOOP ERROR — releasing mouse and stopping: {e}")
    finally:
        try:
            if engine is not None:
                engine.release()
        except Exception:
            pass
        _release_mouse()
        _stop_event.set()


def _loop_running() -> bool:
    t = _loop_thread
    return t is not None and t.is_alive()


def _start_loop() -> bool:
    """Start the control loop (idempotent). Returns True if a loop is running
    when we return (freshly started or already alive)."""
    global _loop_thread, _engine
    with _thread_lock:
        if _loop_running():
            return True
        _stop_event.clear()
        _engine = AirControlEngine() if _HAS_ENGINE else None
        if _engine is None:
            return False
        t = threading.Thread(target=_loop, daemon=True, name=_THREAD_NAME)
        _loop_thread = t
        t.start()
        return True


def _stop_loop() -> bool:
    """Signal the loop to stop and release the mouse NOW (don't wait for the
    thread to notice — the button must never stay held while we join). Returns
    True if a loop was running."""
    with _thread_lock:
        was = _loop_running()
        _stop_event.set()
        try:
            if _engine is not None:
                _engine.release()
        except Exception:
            pass
        _release_mouse()
        return was


# ─── voice actions ─────────────────────────────────────────────────────────
def air_control_on(_: str = "") -> str:
    """'air control on' / 'let me control the mouse with my hand'. Starts the
    loop — an explicit voice command is consent, so this works even while the
    AIR_CONTROL_ENABLED knob is False (the knob only gates auto-start at load)."""
    if not _HAS_ENGINE:
        return "Air control isn't available, sir — the engine didn't load."
    if _is_staging():
        return "Not while I'm in staging, sir."
    ready, why = _sensor_ready()
    if not ready:
        return f"Kinect isn't available, sir — {why}."
    if _loop_running():
        return "Air control is already on, sir — reach a hand toward me to take the cursor."
    if not _start_loop():
        return "I couldn't start air control, sir."
    return ("Air control on, sir. Reach a hand out toward me to take the "
            "cursor, close your fist to grab and drag, a quick squeeze to "
            "click, point to scroll, and drop your hand to let go.")


def air_control_off(_: str = "") -> str:
    """'air control off' / 'hand mouse off'. Stops the loop and force-releases
    any held button either way (safe no-op when nothing was held)."""
    was = _stop_loop()
    if not was:
        return "Air control is already off, sir."
    return "Air control off, sir — the mouse is all yours."


def air_control_status(_: str = "") -> str:
    """'air control status' / 'is air control on'. One finished sentence."""
    if not _HAS_ENGINE:
        return "Air control isn't available, sir — the engine didn't load."
    if not _loop_running():
        ready, why = _sensor_ready()
        tail = "" if ready else f" Note {why}, so it couldn't start right now anyway."
        return f"Air control is off, sir — say 'air control on' to take the cursor by hand.{tail}"
    eng = _engine
    if eng is not None and eng.engaged:
        hand = eng.active_side or "a"
        doing = "dragging" if eng.button_down else "moving the cursor"
        return f"Air control is on and your {hand} hand is {doing}, sir."
    return ("Air control is on, sir, but no hand is engaged — reach one out "
            "toward me, above your waist, to take the cursor.")


# ─── registration ──────────────────────────────────────────────────────────
def register(actions):
    actions["air_control_on"] = air_control_on
    actions["air_control_off"] = air_control_off
    actions["air_control_status"] = air_control_status

    if not _HAS_ENGINE:
        print("  [air-control] engine missing — actions reply gracefully; "
              "loop disabled.")
        return

    # AUTO-START only when the owner opted in via AIR_CONTROL_ENABLED (default
    # False — see core/config.py: a Kinect glitch must never drive the mouse
    # uninvited at boot). The voice action can still start it later.
    if _cfg_flag("AIR_CONTROL_ENABLED") and not _is_staging():
        ready, why = _sensor_ready()
        if ready:
            _start_loop()
            print(f"  [air-control] loop auto-started "
                  f"(AIR_CONTROL_ENABLED, ~{AIR_CONTROL_POLL_HZ:.0f} Hz)")
        else:
            print(f"  [air-control] enabled but not starting: {why}")
    else:
        print("  [air-control] loaded (loop off — say 'air control on' to start; "
              "AIR_CONTROL_ENABLED auto-start is off by default)")
