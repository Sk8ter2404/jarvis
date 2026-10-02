"""
clap_trigger skill — DOUBLE-CLAP to run a routine (off by default).

Two sharp claps ~0.15-0.7 s apart, with nothing else loud around them, run the
"clap routine". Out of the box that is only a spoken "You rang, sir?": the
click-versus-clap test is close to the line for some sounds (a mechanical
key's multi-part click, a pen tap), so a false double clap must cost a line,
never the desk. Once the owner has heard it answer only his claps, "clap
trigger runs the morning setup" (clap_trigger_routine) or Settings switches the
routine to the morning workspace setup (skills/morning_handoff.py's
predictive_morning_setup — Chrome with Apple Music, Teams, the master volume)
or the morning briefing.

HOW IT LISTENS — it never opens a microphone
============================================
A worker thread registers a bounded queue with the main loop's capture fan-out
(bobert_companion.add_record_tap) and feeds every frame record_speech already
captures to core/clap_detector.ClapDetector. No second InputStream (the WASAPI
double-open stall), nothing added to the PortAudio callback beyond the one
``put_nowait`` the fan-out does per tap. Frames arrive only while record_speech
listens, which is also when the clap matters; when the stream pauses (a turn
running, JARVIS speaking on the main thread) the worker sees the gap and
resets the detector, so claps either side of a pause can never pair up. The
worker exists only while CLAP_TRIGGER_ENABLED is on.

WHEN A DOUBLE CLAP IS IGNORED (handle_double_clap's gates, in order)
  * the flag is off, or this is the staging instance;
  * the cool-down (CLAP_TRIGGER_COOLDOWN_S, 60 s) since the last routine;
  * JARVIS was speaking during, or just before, the claps — the playback flag
    and core/self_echo's playback registry (his own voice through the
    speakers, a "clap" in a line, must never trigger anything);
  * Mute Mic is on;
  * night: inside the quiet hours (PHONE_PING_QUIET_START-END, 23:00-07:00 by
    default — the one night window the phone pings use too). A false double
    clap at 3 a.m. must not wake him, set up the desk or speak;
  * he is asleep / in standby — unless CLAP_TRIGGER_WAKE ("clap to wake") is
    on: then the claps wake him (the monolith's _force_wake, the tray's own
    wake) and the routine runs;
  * focus mode (either kind: "focus mode on", or the automatic one a CAD /
    slicer window turns on) — a clap must not move the cursor mid-work;
  * game mode is engaged, or sustained music is playing in the room
    (skills/standby_audio_detect): gunshots and drum hits are clap-shaped;
  * media is playing: the Windows media session (a video with claps in it,
    Apple Music, YouTube), music JARVIS started in the last
    _JARVIS_MUSIC_WINDOW_S, the camera's TV detector (the monolith's
    _ambient_media_is_playing);
  * the speakers are playing anything at all (the playback device's peak
    meter): a game, a call, a clip — the claps may have come out of them.

THE ROUTINE
  CLAP_TRIGGER_ACTION is one of an ALLOW-LIST (_CLAP_SAFE_ACTIONS: the
  workspace setup and its aliases, the morning briefing), run with no argument
  on a short-lived thread; its result is queued as speech (proactive_announce,
  source "clap") unless the action speaks for itself. "acknowledge" (the
  default; also blank) = only the acknowledgement. A clap is ambiguous —
  anyone in the room can clap, and a key or a knock can pass for one — so any
  other name is refused out loud, whatever the setting says, and even an
  allow-listed name is refused if core/action_risk ever classes it as risky
  (_routine_refusal). An alias counts only when it is bound to the SAME
  handler as an allow-listed action.

Voice actions (results spoken verbatim — SPEAK_VERBATIM_ACTIONS):
  clap_trigger_on      — "turn on the clap trigger": live + persisted.
  clap_trigger_off     — "clap trigger off".
  clap_trigger_status  — on/off, what it runs, why the last double clap was
                         ignored, how loud the last clap was vs the threshold.
  clap_trigger_routine — "clap trigger runs the morning setup" / "...the
                         briefing" / "...just answers": CLAP_TRIGGER_ACTION,
                         live + persisted, allow-listed names only.
"""
from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time

try:
    from core.clap_detector import ClapDetector, DEFAULT_MIN_PEAK
except Exception:   # pragma: no cover - core is in-tree; degrade to no detector
    ClapDetector = None
    DEFAULT_MIN_PEAK = 0.12


SPEAK_VERBATIM_ACTIONS = ("clap_trigger_on", "clap_trigger_off",
                          "clap_trigger_status", "clap_trigger_routine")

# "acknowledge" by default (2026-10-02 review): a mechanical key's multi-part
# click passed the detector's click test in a synthetic replay, so until the
# owner has heard it answer only his claps, a false double clap costs one
# spoken line — not a new Chrome window, Teams and the volume at 30 %.
DEFAULT_ACTION = "acknowledge"
WORKSPACE_ACTION = "predictive_morning_setup"
BRIEFING_ACTION = "morning_briefing"
ACK_LINE = "You rang, sir?"
_ACK_NAMES = frozenset({"ack", "acknowledge", "acknowledgement", "none",
                        "off", "nothing"})
_THREAD_NAME = "clap-trigger"
_TAP_QUEUE_MAX = 64          # ~4 s of 64 ms chunks; a stalled worker drops
_GAP_S = 0.25                # no frame for this long = the stream paused
_ECHO_TAIL_S = 1.0           # a line that ended this recently blocks a clap
_CFG_REFRESH_S = 1.0
# Music JARVIS started this recently counts as playing even before (or
# without) the OS media session reporting it.
_JARVIS_MUSIC_WINDOW_S = 600.0
# The playback device's peak meter: anything at or above this (about -34
# dBFS) is sound coming out of the speakers. Read up to 8 times over ~0.3 s
# (stopping at the first loud read), so a quiet passage of a song or a gap
# between two words of a video does not read as silence — measured on the
# owner's PC with music on, single reads dipped to 0.004 between 0.1-0.27.
_OUTPUT_PEAK_MIN = 0.02
_OUTPUT_READS = 8
_OUTPUT_READ_GAP_S = 0.04

# The ONLY actions a clap may run (2026-10-02 review: a hand-written list of
# risky words let guard_off, resume_print, export_memory, text_my_phone, type,
# hotkey, set_model and more through). Anything else is refused out loud,
# whatever CLAP_TRIGGER_ACTION says; an alias counts only when it is bound to
# the same handler as one of these (_routine_refusal).
_CLAP_SAFE_ACTIONS = frozenset({
    WORKSPACE_ACTION, "setup_workspace", "workspace_setup",
    BRIEFING_ACTION,
})
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# ── runtime state ────────────────────────────────────────────────────────
_lock = threading.Lock()
_worker = [None]                   # the worker Thread
_stop_evt = [threading.Event()]
_actions_ref = [None]              # the ACTIONS dict register() was handed
_detector_ref = [None]             # the live detector (status read-outs)


def _fresh_state() -> dict:
    return {"last_fire": None, "last_ignored": None, "last_ignored_at": None,
            "doubles": 0, "fired": 0, "listening": False, "error": None,
            "claps_heard": 0}


_state = _fresh_state()


def _reset_state_for_tests() -> None:
    with _lock:
        _state.clear()
        _state.update(_fresh_state())
    _detector_ref[0] = None


def status_snapshot() -> dict:
    """A copy of the runtime counters (plus the live detector's)."""
    with _lock:
        snap = dict(_state)
    det = _detector_ref[0]
    if det is not None:
        try:
            snap["claps_heard"] = int(det.stats.get("claps", 0))
            snap["last_clap_peak"] = float(det.last_clap_peak)
            snap["last_transient_peak"] = float(det.last_transient_peak)
            snap["last_rejection"] = det.last_rejection
        except Exception:
            pass
    return snap


# ── seams ────────────────────────────────────────────────────────────────

def _bc():
    """The live monolith (by name, or __main__ when it is the entry script)."""
    mod = sys.modules.get("bobert_companion")
    if mod is not None:
        return mod
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "add_record_tap"):
        return main
    return None


def _cfg(name: str, default=None):
    """A live core.config value (read fresh: the voice toggle flips it)."""
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _cfg_bool(name: str, default: bool = False) -> bool:
    return bool(_cfg(name, default))


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except Exception:
        return float(default)


def _is_staging() -> bool:
    """True on the staging / blue-green candidate: the clap trigger never
    listens there (its mic belongs to the live instance)."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    fn = getattr(bc, "_is_staging", None) if bc is not None else None
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _enabled() -> bool:
    return _cfg_bool("CLAP_TRIGGER_ENABLED") and not _is_staging()


def _flag(bc, name: str) -> bool:
    """bc.<name>[0] for the monolith's single-element state cells."""
    try:
        cell = getattr(bc, name, None)
        return bool(cell[0]) if cell is not None else False
    except Exception:
        return False


def _spawn(fn, *args) -> None:
    """Run the routine off the detector's thread (a slow app launch must never
    stall frame consumption). Tests replace this with a direct call."""
    threading.Thread(target=fn, args=args, daemon=True,
                     name="clap-routine").start()


def _note_ignored(reason: str, now: float) -> str:
    with _lock:
        _state["last_ignored"] = reason
        _state["last_ignored_at"] = now
    print(f"  [clap] double clap ignored — {reason}")
    return f"ignored: {reason}"


# ── gates ────────────────────────────────────────────────────────────────

def _speaking_near(bc, t_first: float, t_second: float, now: float) -> bool:
    """JARVIS's own voice was playing during the claps or ended within
    _ECHO_TAIL_S before the first one. Both clocks are the self-echo clock
    (time.monotonic)."""
    if _flag(bc, "_tts_playback_active"):
        return True
    try:
        from core import self_echo as _se
    except Exception:
        return False
    try:
        if _se.playback_live():
            return True
        hit = _se.capture_overlap(t_first - _ECHO_TAIL_S, t_first, t_second,
                                  tail_s=_ECHO_TAIL_S, at=now)
        return hit is not None
    except Exception:
        return False


def _game_mode_active() -> bool:
    try:
        sk = sys.modules.get("skill_game_mode")
        st = getattr(sk, "_st", None) if sk is not None else None
        return bool(getattr(st, "active", False))
    except Exception:
        return False


def _music_playing() -> bool:
    try:
        sk = sys.modules.get("skill_standby_audio_detect")
        fn = getattr(sk, "is_music_currently_playing", None) if sk else None
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _now_local():
    """Local wall-clock time (a seam: tests pin the hour)."""
    import datetime as _dt
    return _dt.datetime.now()


def _night_hours(now_w=None) -> bool:
    """Inside the quiet hours (PHONE_PING_QUIET_START-END, read live; the
    one night window the phone pings use). A window whose start equals its
    end is no window. Never raises."""
    try:
        from core.phone_ping import parse_hhmm
        start = parse_hhmm(_cfg("PHONE_PING_QUIET_START", "23:00"), "23:00")
        end = parse_hhmm(_cfg("PHONE_PING_QUIET_END", "07:00"), "07:00")
        if start == end:
            return False
        now_w = now_w or _now_local()
        m = now_w.hour * 60 + now_w.minute
        if start < end:
            return start <= m < end
        return m >= start or m < end
    except Exception:
        return False


def _focus_active(bc) -> bool:
    """Either focus mode, the same two sources core/phone_ping reads: the
    monolith's focus_mode_active() (honoured while FOCUS_MODE_ENABLED, like
    proactive_announce) and skills/dnd_focus_mode, which switches itself on
    while a CAD / slicer window is open. Never raises."""
    try:
        enabled = _cfg("FOCUS_MODE_ENABLED", True)
        fn = getattr(bc, "focus_mode_active", None) if bc is not None else None
        if (enabled is None or bool(enabled)) and callable(fn) and fn():
            return True
    except Exception:
        pass
    try:
        mod = sys.modules.get("skill_dnd_focus_mode")
        fn = getattr(mod, "is_focus_mode_active", None) if mod is not None else None
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _media_playing(bc) -> bool:
    """Media the claps may have come from: music JARVIS started within
    _JARVIS_MUSIC_WINDOW_S, or the monolith's _ambient_media_is_playing (the
    Windows media session — a YouTube video, Apple Music — the room-music
    detector, the camera's TV detector). Never raises."""
    try:
        cell = getattr(bc, "_jarvis_played_music_at", None)
        played = float(cell[0] or 0.0) if cell else 0.0
        if played > 0.0 and time.time() - played < _JARVIS_MUSIC_WINDOW_S:
            return True
    except Exception:
        pass
    try:
        fn = getattr(bc, "_ambient_media_is_playing", None)
        if callable(fn):
            return bool(fn())
        fn = getattr(bc, "_smtc_media_playing", None)
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


_com_ready = threading.local()


def _speaker_peak() -> "float | None":
    """The default playback device's peak level (0..1), the highest of
    _OUTPUT_READS reads _OUTPUT_READ_GAP_S apart — or None when it cannot be
    read (not Windows, no pycaw / comtypes, no device). Called only for a
    double clap that passed every other gate, on the clap worker's thread:
    never on the audio callback. Never raises."""
    if sys.platform != "win32":
        return None
    try:
        import comtypes
        from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
    except Exception:
        return None
    try:
        if not getattr(_com_ready, "ok", False):
            try:
                comtypes.CoInitialize()
            except Exception:
                pass
            _com_ready.ok = True
        dev = AudioUtilities.GetSpeakers()
        raw = getattr(dev, "_dev", dev)      # pycaw's AudioDevice wrapper
        iface = raw.Activate(IAudioMeterInformation._iid_,
                             comtypes.CLSCTX_ALL, None)
        meter = iface.QueryInterface(IAudioMeterInformation)
        peak = 0.0
        for i in range(_OUTPUT_READS):
            peak = max(peak, float(meter.GetPeakValue()))
            if peak >= _OUTPUT_PEAK_MIN:
                break
            if i + 1 < _OUTPUT_READS:
                time.sleep(_OUTPUT_READ_GAP_S)
        return peak
    except Exception:
        return None


def _speaker_output_active() -> bool:
    """True when the speakers are playing something right now. An unreadable
    meter is not proof of sound: False."""
    peak = _speaker_peak()
    return peak is not None and peak >= _OUTPUT_PEAK_MIN


def handle_double_clap(bc, event: dict, now: "float | None" = None) -> str:
    """Gate one double-clap event from the detector and, if it passes, start
    the routine. Returns "fired" or "ignored: <reason>" (logged + kept for the
    status read-out). ``event`` times are stream seconds; ``now`` is the
    monotonic time the event was produced (≈ its t_detect). Never raises."""
    if now is None:
        now = time.monotonic()
    try:
        with _lock:
            _state["doubles"] += 1
        if not _cfg_bool("CLAP_TRIGGER_ENABLED"):
            return _note_ignored("the clap trigger is off", now)
        if _is_staging():
            return _note_ignored("staging instance", now)
        cooldown = max(0.0, _cfg_float("CLAP_TRIGGER_COOLDOWN_S", 60.0))
        with _lock:
            last = _state["last_fire"]
        if last is not None and now - last < cooldown:
            return _note_ignored(
                f"cool-down ({cooldown - (now - last):.0f} s left)", now)
        t_detect = float(event.get("t_detect", 0.0))
        t1 = now - (t_detect - float(event.get("t_first", t_detect)))
        t2 = now - (t_detect - float(event.get("t_second", t_detect)))
        if _speaking_near(bc, t1, t2, now):
            return _note_ignored("I was speaking", now)
        if _flag(bc, "_mic_muted"):
            return _note_ignored("the mic is muted", now)
        if _night_hours():
            return _note_ignored("it is the middle of the night (quiet "
                                 "hours)", now)
        wake = False
        if _flag(bc, "_sleep_mode") or _flag(bc, "_standby_mode"):
            if not _cfg_bool("CLAP_TRIGGER_WAKE"):
                return _note_ignored("I was asleep (clap to wake is off)", now)
            wake = True
        if _focus_active(bc):
            return _note_ignored("focus mode is on", now)
        if _game_mode_active():
            return _note_ignored("game mode is on", now)
        if _music_playing():
            return _note_ignored("music is playing", now)
        if _media_playing(bc):
            return _note_ignored("media is playing", now)
        if _speaker_output_active():
            return _note_ignored("the speakers were playing something", now)
        with _lock:
            _state["last_fire"] = now
            _state["fired"] += 1
            _state["last_ignored"] = None    # the latest double clap ran
        print(f"  [clap] double clap ({event.get('interval_s', 0):.2f} s "
              f"apart) — running the clap routine"
              + (" after waking" if wake else ""))
        _spawn(_run_routine, bc, wake)
        return "fired"
    except Exception as exc:
        return _note_ignored(f"internal error {type(exc).__name__}", now)


# ── the routine ──────────────────────────────────────────────────────────

def _registry(bc) -> dict:
    reg = getattr(bc, "ACTIONS", None) if bc is not None else None
    if isinstance(reg, dict):
        return reg
    return _actions_ref[0] if isinstance(_actions_ref[0], dict) else {}


def _risk_reasons(name: str) -> tuple:
    """core/action_risk's reasons for ``name`` (the ONE risk classification),
    () when none or when it cannot be read."""
    try:
        from core.action_risk import confirm_reasons
        return tuple(confirm_reasons(name) or ())
    except Exception:
        return ()


def _safe_alias_of(name: str, bc) -> "str | None":
    """The allow-listed action ``name`` is an alias of — bound to the very
    same handler in the live registry — or None."""
    reg = _registry(bc)
    fn = reg.get(name)
    if not callable(fn):
        return None
    for safe in sorted(_CLAP_SAFE_ACTIONS):
        if reg.get(safe) is fn:
            return safe
    return None


def _routine_refusal(name: str, bc) -> "str | None":
    """Why ``name`` must never run on a clap, or None when it may.

    An allow-list (_CLAP_SAFE_ACTIONS, plus an alias bound to the same
    handler as one of them), with core/action_risk as a floor under it: a
    name it classes as stopping JARVIS, sending, deleting, running code,
    acting on the desktop or spending is refused even if it was listed."""
    n = (name or "").strip().lower()
    if not _NAME_RE.match(n):
        return "it is not an action name"
    if n not in _CLAP_SAFE_ACTIONS and _safe_alias_of(n, bc) is None:
        return ("a clap only runs the workspace setup or the morning "
                "briefing")
    if _risk_reasons(n):
        return "it is not safe to run on a clap"
    try:
        if n in set(getattr(bc, "_DESTRUCTIVE_REPLAY_ACTIONS", ()) or ()):
            return "it is a destructive action"
    except Exception:
        pass
    return None


def _configured_action() -> str:
    raw = _cfg("CLAP_TRIGGER_ACTION", DEFAULT_ACTION)
    name = str(raw or "").strip().lower()
    return name or DEFAULT_ACTION


def _announce(bc, text: str) -> None:
    if not text:
        return
    try:
        fn = getattr(bc, "proactive_announce", None)
        if callable(fn):
            fn(text, source="clap")
            return
        fn = getattr(bc, "_speak", None)
        if callable(fn):
            fn(text)
    except Exception as exc:
        print(f"  [clap] could not queue speech: {type(exc).__name__}")


def _wake(bc) -> None:
    """Wake the way the tray's "Wake" does: the monolith's _force_wake (sleep
    and standby cleared under the auto-engage lock, the wake bookkeeping the
    morning chain watches, the overnight flag removed so the overnight engine
    does not restart into an upgrade while he works). Silent: the routine
    does the talking. An older monolith without the helper gets the flags
    cleared under the lock."""
    fw = getattr(bc, "_force_wake", None)
    if callable(fw):
        try:
            fw(speak=False, source="clap")
            print("  [clap] double clap woke me from standby")
            return
        except Exception as exc:
            print(f"  [clap] wake failed: {type(exc).__name__}: {exc}")
            return
    lock = getattr(bc, "_standby_auto_engage_lock", None)
    try:
        if lock is not None:
            with lock:
                bc._sleep_mode[0] = False
                bc._standby_mode[0] = False
        else:
            bc._sleep_mode[0] = False
            bc._standby_mode[0] = False
        try:
            bc._write_hud_state(sleep_mode=False, standby_mode=False,
                                state="Idle")
        except Exception:
            pass
        print("  [clap] double clap woke me from standby")
    except Exception as exc:
        print(f"  [clap] wake failed: {type(exc).__name__}: {exc}")


def _run_routine(bc, wake: bool = False) -> None:
    """Wake if asked, then run the configured routine and queue what it says.
    Never raises (it runs on its own daemon thread)."""
    try:
        if wake:
            _wake(bc)
        name = _configured_action()
        if name in _ACK_NAMES:
            _announce(bc, ACK_LINE)
            return
        refusal = _routine_refusal(name, bc)
        if refusal:
            print(f"  [clap] refusing clap routine {name!r}: {refusal}")
            _announce(bc, f"I heard the claps, sir, but I won't run {name} on "
                          f"a clap — {refusal}.")
            return
        fn = _registry(bc).get(name)
        if not callable(fn):
            print(f"  [clap] clap routine {name!r} is not a registered "
                  f"action — acknowledging instead")
            _announce(bc, ACK_LINE)
            return
        try:
            result = fn("")
        except Exception as exc:
            print(f"  [clap] clap routine {name!r} raised "
                  f"{type(exc).__name__}: {exc}")
            _announce(bc, "Your clap routine hit a snag, sir.")
            return
        try:
            voiced = getattr(bc, "is_self_voiced", None)
            if callable(voiced) and voiced(name):
                return
        except Exception:
            pass
        text = result.strip() if isinstance(result, str) else ""
        _announce(bc, text or ACK_LINE)
    except Exception as exc:
        print(f"  [clap] routine failed: {type(exc).__name__}: {exc}")


# ── the worker: the mic tap → the detector ────────────────────────────────

def _sample_rate(bc) -> int:
    try:
        cell = getattr(bc, "_record_speech_sr", None)
        if cell is not None and int(cell[0]) > 0:
            return int(cell[0])
    except Exception:
        pass
    try:
        return int(getattr(bc, "SAMPLE_RATE", 16000) or 16000)
    except Exception:
        return 16000


def _worker_loop(bc, stop_evt: threading.Event) -> None:  # noqa: C901
    add = getattr(bc, "add_record_tap", None)
    remove = getattr(bc, "remove_record_tap", None)
    tap: "queue.Queue" = queue.Queue(maxsize=_TAP_QUEUE_MAX)
    try:
        add(tap)
    except Exception as exc:
        with _lock:
            _state["error"] = f"could not tap the mic: {type(exc).__name__}"
        print(f"  [clap] {_state['error']}")
        return
    with _lock:
        _state["listening"] = True
        _state["error"] = None
    print("  [clap] listening for a double clap (sharing the main loop's mic)")
    det = None
    gap = True
    next_cfg = 0.0
    min_peak = DEFAULT_MIN_PEAK
    sr = 16000
    try:
        while not stop_evt.is_set():
            if not _enabled():
                break
            try:
                frame = tap.get(timeout=_GAP_S)
            except queue.Empty:
                gap = True              # record_speech paused: a hole in time
                continue
            if tap.qsize() >= _TAP_QUEUE_MAX - 1:
                # Frames were dropped behind us: what is queued is stale and
                # discontinuous with what comes next. Start clean.
                try:
                    while True:
                        tap.get_nowait()
                except queue.Empty:
                    pass
                gap = True
                continue
            now = time.monotonic()
            if now >= next_cfg:
                next_cfg = now + _CFG_REFRESH_S
                sr = _sample_rate(bc)
                min_peak = _cfg_float("CLAP_TRIGGER_MIN_PEAK", DEFAULT_MIN_PEAK)
            if det is None or det.sample_rate != sr:
                det = ClapDetector(sr, min_peak=min_peak)
                _detector_ref[0] = det
                gap = False
            det.min_peak = min_peak
            if gap:
                det.reset()
                gap = False
            for ev in det.feed(frame):
                handle_double_clap(bc, ev, now=time.monotonic())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [clap] worker stopped on {type(exc).__name__}: {exc}")
    finally:
        try:
            remove(tap)
        except Exception:
            pass
        with _lock:
            _state["listening"] = False
        print("  [clap] stopped listening for claps")


def _worker_alive() -> bool:
    t = _worker[0]
    return t is not None and t.is_alive()


def _start_worker() -> bool:
    """Start listening (idempotent). False when it cannot: staging, no
    detector, or no monolith tap API."""
    if _is_staging() or ClapDetector is None:
        return False
    bc = _bc()
    if (bc is None or not callable(getattr(bc, "add_record_tap", None))
            or not callable(getattr(bc, "remove_record_tap", None))):
        with _lock:
            _state["error"] = "the main loop's mic tap is not available"
        return False
    old = _worker[0]
    if old is not None and old.is_alive() and _stop_evt[0].is_set():
        # A stop is in flight (its tap removal is bounded by _GAP_S): let it
        # finish so a quick off -> on never leaves two workers or none.
        old.join(timeout=2.0 * _GAP_S + 0.5)
    with _lock:
        if _worker[0] is not None and _worker[0].is_alive():
            return True
        evt = threading.Event()
        _stop_evt[0] = evt
        t = threading.Thread(target=_worker_loop, args=(bc, evt), daemon=True,
                             name=_THREAD_NAME)
        _worker[0] = t
    t.start()
    return True


def _stop_worker(timeout: float = 2.0) -> None:
    """Stop listening; waits (bounded) for the tap to be removed."""
    _stop_evt[0].set()
    t = _worker[0]
    if t is not None and t is not threading.current_thread():
        t.join(timeout=max(0.0, timeout))


# ── persistence (the Settings writer every voice toggle uses) ─────────────

def _persist_setting(key: str, value) -> bool:
    try:
        from tools import settings_window as sw
    except Exception:
        return False
    try:
        current = sw.load_settings()
        if not isinstance(current, dict):
            current = {}
        current[key] = value
        sw.save_settings(current, changed=(key,))
        return True
    except Exception:
        return False


def _set_enabled(on: bool) -> bool:
    try:
        import core.config as _c
        _c.CLAP_TRIGGER_ENABLED = bool(on)
    except Exception:
        pass
    return _persist_setting("CLAP_TRIGGER_ENABLED", bool(on))


def _routine_phrase() -> str:
    name = _configured_action()
    if name in _ACK_NAMES:
        return "I'll answer"
    if name in (WORKSPACE_ACTION, "setup_workspace", "workspace_setup"):
        return "I'll set up your workspace"
    if name == BRIEFING_ACTION:
        return "I'll give you the morning briefing"
    return f"I'll run {name}"


_TRIAL_HINT = (" For now a double clap only gets a 'You rang, sir?', so we can "
               "hear what else sets it off; once it only answers you, say "
               "'clap trigger runs the morning setup'.")


# ── actions ──────────────────────────────────────────────────────────────

def clap_trigger_on(_: str = "") -> str:
    """Turn the clap trigger on (live + persisted) and start listening."""
    if _is_staging():
        return "The clap trigger stays off on the staging instance, sir."
    already = _cfg_bool("CLAP_TRIGGER_ENABLED") and _worker_alive()
    persisted = _set_enabled(True)
    listening = _start_worker()
    if already:
        msg = "The clap trigger is already on, sir."
    elif _configured_action() in _ACK_NAMES:
        msg = "Clap trigger on, sir." + _TRIAL_HINT
    else:
        msg = (f"Clap trigger on, sir — clap twice and {_routine_phrase()}.")
    if not listening:
        msg += " I can't reach the microphone feed yet, so it starts when I do."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


def clap_trigger_off(_: str = "") -> str:
    """Turn the clap trigger off (live + persisted) and stop listening."""
    was_on = _cfg_bool("CLAP_TRIGGER_ENABLED") or _worker_alive()
    persisted = _set_enabled(False)
    _stop_worker()
    msg = "Clap trigger off, sir." if was_on else \
        "The clap trigger is already off, sir."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


def clap_trigger_status(_: str = "") -> str:
    """On/off, what it runs, and what the last claps did."""
    if not _cfg_bool("CLAP_TRIGGER_ENABLED"):
        return ("The clap trigger is off, sir — say 'turn on the clap "
                "trigger' and a double clap will run your clap routine.")
    snap = status_snapshot()
    parts = [f"The clap trigger is on, sir: clap twice and {_routine_phrase()}"]
    if _cfg_bool("CLAP_TRIGGER_WAKE"):
        parts[0] += ", and it wakes me from standby"
    parts[0] += "."
    name = _configured_action()
    if name not in _ACK_NAMES and _routine_refusal(name, _bc()):
        parts.append(f"The routine in Settings, {name}, is not one a clap may "
                     f"run, so I only answer.")
    if not snap.get("listening") and not _worker_alive():
        parts.append("I'm not hearing the microphone feed right now.")
    if snap.get("last_ignored"):
        parts.append(f"I ignored the last double clap because "
                     f"{snap['last_ignored']}.")
    thr = _cfg_float("CLAP_TRIGGER_MIN_PEAK", DEFAULT_MIN_PEAK)
    peak = snap.get("last_clap_peak")
    sharp = snap.get("last_transient_peak")
    if peak:
        parts.append(f"The last clap I heard peaked at {peak:.2f}; the "
                     f"threshold is {thr:.2f}.")
    elif sharp and sharp < thr:
        parts.append(f"I haven't heard a clap loud enough yet: the last sharp "
                     f"sound peaked at {sharp:.2f}, under the {thr:.2f} "
                     f"threshold — lower it in Settings if that was you.")
    return " ".join(parts)


_ROUTINE_WORDS = (
    (re.compile(r"\b(?:answer|answers|acknowledge|acknowledges|ack|nothing|"
                r"rang|ring|reply|replies|just talk|just speak)\b"),
     DEFAULT_ACTION),
    (re.compile(r"\bbriefing\b"), BRIEFING_ACTION),
    (re.compile(r"\b(?:morning|workspace|work space|desk|setup|set up|"
                r"routine|workshop)\b"), WORKSPACE_ACTION),
)


def clap_trigger_routine(arg: str = "") -> str:
    """Choose what a double clap runs (CLAP_TRIGGER_ACTION, live + saved):
    the workspace setup, the morning briefing, or just the acknowledgement.
    Only the allow-listed routines — any other request is refused."""
    text = " ".join(str(arg or "").lower().replace("-", " ").split())
    want = None
    for rx, name in _ROUTINE_WORDS:
        if rx.search(text):
            want = name
            break
    if want is None:
        return ("A clap can set up your workspace, give you the morning "
                "briefing, or just answer, sir — which would you like?")
    try:
        import core.config as _c
        _c.CLAP_TRIGGER_ACTION = want
    except Exception:
        pass
    persisted = _persist_setting("CLAP_TRIGGER_ACTION", want)
    if want == DEFAULT_ACTION:
        msg = "Done, sir — a double clap just gets a 'You rang, sir?' now."
    elif want == BRIEFING_ACTION:
        msg = "Done, sir — a double clap gives you the morning briefing now."
    else:
        msg = "Done, sir — a double clap sets up your workspace now."
    if not _cfg_bool("CLAP_TRIGGER_ENABLED"):
        msg += " The clap trigger itself is off: say 'turn on the clap trigger'."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


# ── utterance route: claim the exact on/off/status requests before the LLM ──
_NOUN = (r"(?:the )?(?:double ?)?clap(?:ping)? "
         r"(?:trigger|detection|detector|sensor|switch)")
_LEAD = r"^(?:(?:hey |ok |okay )?jarvis )?(?:please )?(?:can you |could you )?"
_TAIL = r"(?: now)?(?: please)?$"
_ROUTES = (
    (re.compile(_LEAD + r"(?:is " + _NOUN + r" (?:on|off|enabled|active|"
                r"armed|working|listening)|(?:what is )?" + _NOUN
                + r" status)" + _TAIL), "[ACTION: clap_trigger_status]"),
    (re.compile(_LEAD + r"(?:(?:turn|switch|shut) off " + _NOUN
                + r"|(?:turn|switch|shut) " + _NOUN + r" off|(?:disable|"
                r"deactivate|disarm|stop) " + _NOUN + r"|" + _NOUN + r" off)"
                + _TAIL), "[ACTION: clap_trigger_off]"),
    (re.compile(_LEAD + r"(?:(?:turn|switch) on " + _NOUN
                + r"|(?:turn|switch) " + _NOUN + r" on|(?:enable|activate|"
                r"arm|start) " + _NOUN + r"|" + _NOUN + r" on)" + _TAIL),
     "[ACTION: clap_trigger_on]"),
)

# "clap trigger runs the morning setup" / "make the clap trigger just answer"
# / "set the clap routine to the briefing" -> clap_trigger_routine with the
# routine's canonical name. Only when the rest names an allow-listed routine.
_RNOUN = r"(?:the |my )?(?:double ?)?clap(?:ping)? (?:trigger|routine)"
_ROUTINE_ROUTES = (
    re.compile(_LEAD + r"(?:make |have |let )?" + _RNOUN
               + r"(?: (?:to|should|will))? (?P<rest>(?:runs?|do|does|start|"
               r"starts|gives?|plays?|just|only|sets?) .+?)" + _TAIL),
    re.compile(_LEAD + r"(?:set|change|switch) " + _RNOUN
               + r" (?:to|so it) (?P<rest>.+?)" + _TAIL),
)
_ROUTINE_TOKEN = {DEFAULT_ACTION: "acknowledge",
                  BRIEFING_ACTION: "morning briefing",
                  WORKSPACE_ACTION: "morning setup"}


def _clap_route(text):
    """Utterance route: the clap-trigger token for an exact on / off / status
    / routine request, else None. Never raises."""
    try:
        s = str(text or "").lower().replace("-", " ").replace("what's",
                                                              "what is")
        s = " ".join(re.sub(r"[^a-z0-9' ]+", " ", s).split())
        if "clap" not in s:
            return None
        for rx, token in _ROUTES:
            if rx.match(s):
                return token
        for rx in _ROUTINE_ROUTES:
            m = rx.match(s)
            if not m:
                continue
            for wrx, name in _ROUTINE_WORDS:
                if wrx.search(m.group("rest")):
                    return ("[ACTION: clap_trigger_routine, %s]"
                            % _ROUTINE_TOKEN[name])
            return None
        return None
    except Exception:
        return None


# ── registration ─────────────────────────────────────────────────────────

def register(actions):
    actions["clap_trigger_on"] = clap_trigger_on
    actions["clap_trigger_off"] = clap_trigger_off
    actions["clap_trigger_status"] = clap_trigger_status
    actions["clap_trigger_routine"] = clap_trigger_routine
    _actions_ref[0] = actions
    try:
        su = globals().get("skill_utils") or {}
        reg = su.get("register_utterance_route") if isinstance(su, dict) else None
        if callable(reg):
            reg(_clap_route, "clap trigger")
    except Exception:
        pass
    if _cfg_bool("CLAP_TRIGGER_ENABLED") and not _is_staging():
        if _start_worker():
            print("  [clap] clap trigger on — listening for a double clap")
        else:
            print("  [clap] clap trigger on, but the mic tap is unavailable")
