"""
Guard mode — a multi-angle security coverage array for JARVIS.

JARVIS already has THREE cameras pointed at the workspace: two monitor-mounted
USB webcams (tracked by the monolith's _face_tracking_thread) and a room-facing
Xbox Kinect v2 (the audio/kinect_bridge.py sensor). The other camera skills READ
those feeds for awareness; this skill turns them into an armed SECURITY ARRAY.

When the owner ARMS guard mode (an explicit voice action — never automatic), a
background daemon polls every available camera a few times a second and watches
for movement via frame differencing (grayscale → cv2.absdiff → threshold → the
fraction of changed pixels). The Kinect adds a second, stronger signal: a tracked
skeleton / get_presence().present means a *person* is in the room, not just a
flicker of pixels. A detection that PERSISTS for a couple of frames (debounce, to
shrug off lighting flicker) snapshots the triggering frame to a gitignored folder
and — rate-limited — speaks a proactive alert ("Movement detected on the desk
camera, sir.") and fires the phone-push path if one is configured.

  guard_on    — arm the array + start the monitor daemon. Aliases: 'guard the
                room', 'watch the room', 'arm security', 'keep watch'.
  guard_off   — disarm + stop the daemon. Aliases: 'stand down', 'disarm',
                'stop guarding'.
  guard_status— armed/disarmed, for how long, how many events so far. Aliases:
                'are you watching', 'guard status'.

Design notes
------------
* TWO gates, deliberately separate:
    - core.config.KINECT_GUARD_ENABLED (default False) decides whether arming is
      even ALLOWED — a master opt-in, like every other Kinect capability.
    - the runtime armed/disarmed flag is flipped by the explicit voice action.
  Arming is never automatic; KINECT_GUARD_ENABLED off → guard_on politely
  declines and points at the setting.
* The monitor is a daemon thread that ONLY does work while armed. The per-tick
  function (_guard_tick) is pure-ish — it's handed the camera frames + a wall-
  clock timestamp string — so the test suite drives it directly with mocked
  numpy frames and never sleeps or touches hardware.
* Motion detection (_frame_motion / _is_motion) is a free function over two BGR
  ndarrays; it never reads the clock. Snapshots are named by a timestamp the
  loop passes in, so a test can assert the filename deterministically and the
  snapshot dir is patchable (tests point it at a tmp dir — NEVER the real
  data/ tree).
* Every cv2 / bridge / monolith touch is wrapped so a dark, stale, or absent
  camera is simply skipped, and a missing monolith / Kinect degrades to "no
  signal" instead of raising into the voice loop.
"""
from __future__ import annotations

import os
import sys
import threading
import time


# ─── tunables ────────────────────────────────────────────────────────────
GUARD_POLL_INTERVAL   = 0.25   # seconds between ticks (~4 Hz)
INITIAL_DELAY_SECONDS = 2.0    # let the face-track thread come up first

# Motion: a frame counts as "moved" when the fraction of pixels that changed
# (post-threshold) exceeds MOTION_PIXEL_FRACTION. Frames are downscaled to
# MOTION_RESIZE_WIDTH first so the diff is fast and a little noise-tolerant.
MOTION_RESIZE_WIDTH   = 320
MOTION_DIFF_THRESHOLD = 25     # per-pixel absdiff value to call a pixel "changed"
MOTION_PIXEL_FRACTION = 0.02   # >2% of pixels changed = motion on that camera
# A camera must show motion on this many CONSECUTIVE ticks before it triggers —
# this debounce shrugs off a single-frame lighting flicker / sensor noise spike.
MOTION_DEBOUNCE_FRAMES = 3

# Alerts are rate-limited: at most one spoken/pushed alert per this window, so a
# person moving around for a minute produces ONE alert, not a stream.
GUARD_ALERT_COOLDOWN_SEC = 30.0

# Snapshots/events are ALSO rate-limited, per camera: a continuous presence must
# not write a PNG + bump the event count every 0.25 s tick (~14k files/hour) —
# one trigger per camera per this window is plenty of evidence.
GUARD_TRIGGER_COOLDOWN_SEC = 5.0

# Snapshots land here (gitignored — data/* is ignored). Tests patch this to a
# tmp dir so a test run NEVER writes into the real data/ tree.
_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR  = os.path.join(_PROJECT_DIR, "data", "guard_snapshots")


# ─── runtime state (module-level, thread-safe) ──────────────────────────
# List-wrapped where the loop mutates without `global`. All reads/writes of the
# compound state go under _guard_lock.
_guard_lock = threading.RLock()
_armed       = [False]          # is the array currently armed
_armed_since = [0.0]            # wall-clock ts arming happened (0.0 = disarmed)
_event_count = [0]             # how many detections fired this arming session
_last_event  = [None]          # dict | None: {"camera","ts","kind","distance_m"}
_last_alert_at = [0.0]          # wall-clock ts of the last SPOKEN/PUSHED alert
_monitor_thread = [None]       # type: ignore[var-annotated]

# Per-camera bookkeeping for the loop (keyed by a stable camera label):
#   _prev_frames[label]     -> the previous downscaled grayscale ndarray
#   _motion_streak[label]   -> consecutive ticks of motion seen so far
#   _last_trigger_at[label] -> wall-clock ts of the last snapshot/event trigger
# Reset on every arm so a new session starts clean. Guarded by _guard_lock.
_prev_frames: dict = {}
_motion_streak: dict = {}
_last_trigger_at: dict = {}

# Monotonic arm-generation counter: each guard_on bumps it and hands the value
# to its monitor thread; a thread whose generation is stale exits on its next
# tick. Closes the off→on rearm race where a winding-down thread could be
# mistaken for a live poller. Guarded by _guard_lock.
_monitor_gen = [0]


# ─── seams to the rest of JARVIS ─────────────────────────────────────────

def _bc():
    """The live monolith module (main or by-name), or None. Mirrors the lookup
    every other camera-aware skill uses."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _kinect_bridge():
    """The live kinect_bridge module, or None. Prefer the instance the monolith
    already imported; fall back to a direct import so the skill works even when
    the monolith hasn't loaded it."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _phone_bridge():
    """The live phone_bridge skill module (registered as skill_phone_bridge by
    the loader), or None. Used for the optional push path; absent → no push."""
    return sys.modules.get("skill_phone_bridge")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence (early boot
    / standalone test). Read fresh each call so a Settings toggle takes effect
    without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — guard mode must never fire an alert
    there. Matches the monolith's own gate plus the raw env var so the check
    holds even before the monolith is importable."""
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


# ─── frame collection (reuses the shared caches + the Kinect bridge) ──────

def _collect_frames() -> list[tuple[str, object]]:
    """Grab the most recent BGR frame from EACH available source as a list of
    (camera_label, bgr_ndarray). Webcams come from the monolith's shared
    _camera_latest_frame cache (copied under the lock); the Kinect via the
    bridge's get_color_bgr(). Missing sources are simply omitted. NEVER raises.

    The label is a stable, spoken-friendly id ("the left monitor camera",
    "the Kinect") used both for the snapshot filename and the alert text, and as
    the per-camera key for previous-frame / debounce bookkeeping."""
    frames: list[tuple[str, object]] = []

    # Webcams: copy each cached frame under the monolith's lock.
    bc = _bc()
    if bc is not None:
        try:
            from core.config import CAMERAS
        except Exception:
            CAMERAS = []
        lock = getattr(bc, "_camera_state_lock", None)
        latest = getattr(bc, "_camera_latest_frame", {}) or {}
        try:
            def _grab():
                for cam in CAMERAS:
                    idx = cam.get("index")
                    fr = latest.get(idx)
                    if fr is None:
                        continue
                    side = "left" if cam.get("look_x", 0.5) < 0.5 else "right"
                    label = f"the {side} monitor camera"
                    try:
                        frames.append((label, fr.copy()))
                    except Exception:
                        frames.append((label, fr))
            if lock is not None:
                with lock:
                    _grab()
            else:
                _grab()
        except Exception:
            pass

    # Kinect: only when it's enabled AND streaming.
    if _cfg_flag("KINECT_ENABLED"):
        kb = _kinect_bridge()
        if kb is not None:
            try:
                ok, _reason = kb.available()
            except Exception:
                ok = False
            if ok:
                try:
                    bgr = kb.get_color_bgr()
                except Exception:
                    bgr = None
                if bgr is not None:
                    frames.append(("the Kinect", bgr))
    return frames


def _kinect_intrusion() -> dict | None:
    """If the Kinect is enabled, streaming, and sees a tracked body, return a
    strong-intrusion descriptor {"present": True, "count": int,
    "nearest_m": float|None}; else None. NEVER raises. This is a SEPARATE, more
    reliable signal than webcam pixel motion — a skeleton means a *person*."""
    if not _cfg_flag("KINECT_ENABLED"):
        return None
    kb = _kinect_bridge()
    if kb is None:
        return None
    try:
        ok, _reason = kb.available()
        if not ok:
            return None
        presence = kb.get_presence() or {}
    except Exception:
        return None
    if not presence.get("present"):
        return None
    return {
        "present": True,
        "count": int(presence.get("count", 0) or 0),
        "nearest_m": presence.get("nearest_m"),
    }


# ─── motion detection (pure functions over frames — no clock, no I/O) ─────

def _prep_gray(frame):
    """Downscale a BGR (or already-gray) ndarray to MOTION_RESIZE_WIDTH and
    return an 8-bit single-channel ndarray for diffing, or None if cv2/numpy
    aren't available or the frame is unusable. NEVER raises."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    try:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 2:
            gray = arr.astype("uint8") if arr.dtype != np.uint8 else arr
        else:
            return None
        h, w = gray.shape[:2]
        if w <= 0 or h <= 0:
            return None
        if w > MOTION_RESIZE_WIDTH:
            scale = MOTION_RESIZE_WIDTH / float(w)
            new_size = (MOTION_RESIZE_WIDTH, max(1, int(round(h * scale))))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        return gray
    except Exception:
        return None


def _frame_motion(prev_gray, cur_gray) -> float:
    """Fraction (0.0–1.0) of pixels that changed between two prepared grayscale
    frames: |absdiff| > MOTION_DIFF_THRESHOLD, counted and divided by the pixel
    total. Returns 0.0 on any mismatch / failure (never raises). Mismatched
    shapes (a camera that changed resolution) read as 0.0 rather than a spurious
    full-frame diff."""
    if prev_gray is None or cur_gray is None:
        return 0.0
    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0
    try:
        if prev_gray.shape != cur_gray.shape:
            return 0.0
        diff = cv2.absdiff(prev_gray, cur_gray)
        changed = int(np.count_nonzero(diff > MOTION_DIFF_THRESHOLD))
        total = int(diff.size)
        if total <= 0:
            return 0.0
        return changed / float(total)
    except Exception:
        return 0.0


def _is_motion(prev_gray, cur_gray) -> bool:
    """True when the changed-pixel fraction exceeds MOTION_PIXEL_FRACTION."""
    return _frame_motion(prev_gray, cur_gray) > MOTION_PIXEL_FRACTION


# ─── snapshot writing (named by a passed-in timestamp; dir is patchable) ──

def _safe_label(label: str) -> str:
    """Turn a spoken camera label into a filename-safe token."""
    keep = []
    for ch in label.strip().lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_"):
            keep.append("_")
    token = "".join(keep).strip("_")
    # Collapse runs of underscores.
    while "__" in token:
        token = token.replace("__", "_")
    return token or "camera"


def _save_snapshot(frame, camera_label: str, ts_label: str) -> str | None:
    """Write `frame` (a BGR ndarray) to SNAPSHOT_DIR/<ts_label>_<camera>.png and
    return the path, or None on any failure. `ts_label` is supplied by the caller
    (the loop) — this function never reads the clock, so a test can assert the
    exact filename. Creates SNAPSHOT_DIR if needed. NEVER raises."""
    if frame is None:
        return None
    try:
        import cv2
    except Exception:
        return None
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        fname = f"{ts_label}_{_safe_label(camera_label)}.png"
        path = os.path.join(SNAPSHOT_DIR, fname)
        ok = cv2.imwrite(path, frame)
        return path if ok else None
    except Exception:
        return None


# ─── alerting (proactive speech + optional phone push, rate-limited) ──────

def _alert_text(camera_label: str, kind: str, distance_m=None) -> str:
    """The spoken/pushed alert line for a detection. `kind` is 'kinect' (a
    tracked person) or 'motion' (webcam pixel motion)."""
    if kind == "kinect":
        if distance_m:
            return (f"Someone is in the room, sir — the Kinect sees a person "
                    f"about {float(distance_m):.1f} metres out.")
        return "Someone is in the room, sir — the Kinect sees a person."
    return f"Movement detected on {camera_label}, sir."


def _speak_alert(message: str) -> bool:
    """Speak a proactive alert via the monolith's proactive_announce queue (the
    same race-safe writer wellness/timers use), with an urgent voice mood when
    available. Best-effort: returns False if the monolith isn't loaded / enqueue
    failed."""
    bc = _bc()
    if bc is None:
        return False
    fn = getattr(bc, "proactive_announce", None)
    if not callable(fn):
        return False
    try:
        # mood is an optional kwarg on proactive_announce; urgent_clipped is the
        # VIP/alert preset. Tolerate an older signature without it.
        try:
            return bool(fn(message, source="guard", mood="urgent_clipped"))
        except TypeError:
            return bool(fn(message, source="guard"))
    except Exception:
        return False


def _push_alert(message: str) -> bool:
    """Fire the optional phone-push path if the phone_bridge skill is loaded and
    a backend is configured. confirm=False — a security alert is fire-and-forget,
    not a draft to read back. Best-effort; returns False when no push happened."""
    pb = _phone_bridge()
    if pb is None:
        return False
    fn = getattr(pb, "push_to_phone", None)
    if not callable(fn):
        return False
    try:
        results = fn(message, priority="urgent", source="guard",
                     title="JARVIS guard", confirm=False)
        return bool(results) and any(results.values())
    except Exception:
        return False


def _fire_alert(camera_label: str, kind: str, now: float,
                distance_m=None) -> bool:
    """Speak + push ONE alert, respecting the cooldown. Returns True if an alert
    actually went out this call, False if suppressed by the cooldown. Updates
    _last_alert_at under the lock. The EVENT count is bumped separately (every
    detection counts toward guard_status even when the alert is rate-limited)."""
    with _guard_lock:
        last = _last_alert_at[0]
        if last and (now - last) < GUARD_ALERT_COOLDOWN_SEC:
            return False
        _last_alert_at[0] = now
    message = _alert_text(camera_label, kind, distance_m)
    # Never actually speak/push on a staging/test instance.
    if _is_staging():
        return True
    spoke = _speak_alert(message)
    _push_alert(message)
    return spoke


# ─── the single monitor tick (driven directly by tests; no sleeping) ──────

def _claim_trigger(camera_label: str, now: float) -> bool:
    """True if `camera_label` may trigger (snapshot + event) at `now`, claiming
    the per-camera trigger-cooldown slot; False while the camera is still inside
    GUARD_TRIGGER_COOLDOWN_SEC of its last trigger. Keeps a continuous presence
    from writing a snapshot + bumping the event count on every single tick."""
    with _guard_lock:
        last = _last_trigger_at.get(camera_label, 0.0)
        if last and (now - last) < GUARD_TRIGGER_COOLDOWN_SEC:
            return False
        _last_trigger_at[camera_label] = now
        return True


def _guard_tick(frames, kinect, ts_label: str, now: float) -> dict:
    """Process ONE poll of the array. Pure-ish: all sensor reads are handed in.

      frames:   list of (camera_label, bgr_ndarray) from _collect_frames()
      kinect:   dict from _kinect_intrusion() (a tracked person) or None
      ts_label: filename-safe wall-clock string (the loop builds it; we never
                read the clock here so a test can assert the snapshot name)
      now:      float wall-clock seconds, for cooldown math (handed in)

    Returns a summary dict:
      {"triggered": [list of camera labels that fired this tick],
       "snapshots": [list of snapshot paths written],
       "alerted":   bool}     # whether an alert actually went out this tick

    Side effects (only when something triggers): writes a snapshot per trigger,
    bumps the event count + last-event info, and fires ONE rate-limited alert.
    Updates per-camera previous-frame + debounce streak bookkeeping. NEVER
    raises into the caller."""
    triggered: list[str] = []
    snapshots: list[str] = []
    alerted = False

    # 1) Kinect intrusion — the strong signal. A tracked body fires immediately
    #    (no pixel-debounce needed: the skeleton tracker already debounces), but
    #    a CONTINUOUS presence re-triggers at most once per trigger cooldown —
    #    otherwise every 0.25 s tick would write a full-res snapshot and bump
    #    the event count (~14k PNGs/hour for one visitor).
    if kinect and kinect.get("present") and _claim_trigger("the Kinect", now):
        label = "the Kinect"
        triggered.append(label)
        # Snapshot the Kinect's own colour frame if we were handed one.
        kin_frame = None
        for lbl, fr in frames:
            if lbl == label:
                kin_frame = fr
                break
        path = _save_snapshot(kin_frame, label, ts_label) if kin_frame is not None else None
        if path:
            snapshots.append(path)
        _record_event(label, "kinect", now, distance_m=kinect.get("nearest_m"))
        if _fire_alert(label, "kinect", now, distance_m=kinect.get("nearest_m")):
            alerted = True

    # 2) Webcam pixel motion — per camera, with debounce. Skip the Kinect's own
    #    colour frame here (it's covered by the skeleton signal above).
    for label, frame in frames:
        if label == "the Kinect":
            continue
        cur = _prep_gray(frame)
        with _guard_lock:
            prev = _prev_frames.get(label)
            _prev_frames[label] = cur if cur is not None else prev
            # First frame for this camera → no previous to diff against, skip.
            if prev is None or cur is None:
                _motion_streak[label] = 0
                continue
            moved = _is_motion(prev, cur)
            if moved:
                _motion_streak[label] = _motion_streak.get(label, 0) + 1
            else:
                _motion_streak[label] = 0
            streak = _motion_streak.get(label, 0)
            fired = streak >= MOTION_DEBOUNCE_FRAMES
            if fired:
                # Reset the streak so a continuous mover re-arms cleanly after
                # the alert cooldown rather than firing every single tick.
                _motion_streak[label] = 0
        # Trigger-cooldown gate: a continuous mover re-fires the debounce every
        # few ticks; cap snapshots + events to one per camera per window.
        if not fired or not _claim_trigger(label, now):
            continue
        triggered.append(label)
        path = _save_snapshot(frame, label, ts_label)
        if path:
            snapshots.append(path)
        _record_event(label, "motion", now)
        if _fire_alert(label, "motion", now):
            alerted = True

    return {"triggered": triggered, "snapshots": snapshots, "alerted": alerted}


def _record_event(camera_label: str, kind: str, now: float,
                  distance_m=None) -> None:
    """Bump the rolling event count + stash last-event info for guard_status.
    Every detection counts here, even when the spoken alert is rate-limited."""
    with _guard_lock:
        _event_count[0] += 1
        _last_event[0] = {
            "camera": camera_label,
            "ts": now,
            "kind": kind,
            "distance_m": distance_m,
        }


# ─── the monitor daemon (only works while armed) ─────────────────────────

def _monitor_loop(gen: int) -> None:  # pragma: no cover - non-terminating daemon; each tick delegates to _guard_tick, which is unit-tested directly
    """Background poller for arm-generation `gen`. Sleeps briefly so the
    face-track thread comes up, then polls every GUARD_POLL_INTERVAL. Exits when
    disarmed OR when a newer arming session has started its own thread (stale
    generation) — so an off→on rearm never leaves two pollers running."""
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        try:
            with _guard_lock:
                armed = _armed[0] and _monitor_gen[0] == gen
            if not armed:
                # Disarmed (or superseded by a newer arm): wind down.
                return
            frames = _collect_frames()
            kinect = _kinect_intrusion()
            now = time.time()
            ts_label = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            # Disambiguate same-second snapshots with a millisecond suffix.
            ts_label = f"{ts_label}_{int((now % 1) * 1000):03d}"
            _guard_tick(frames, kinect, ts_label, now)
        except Exception as e:
            print(f"  [guard] monitor tick error: {e}")
        time.sleep(GUARD_POLL_INTERVAL)


def _reset_session() -> None:
    """Clear per-session bookkeeping so each arm starts clean. Under the lock."""
    with _guard_lock:
        _event_count[0] = 0
        _last_event[0] = None
        _last_alert_at[0] = 0.0
        _prev_frames.clear()
        _motion_streak.clear()
        _last_trigger_at.clear()


# ─── actions ─────────────────────────────────────────────────────────────

def guard_on(_: str = "") -> str:
    """Arm the security array + start the monitor daemon. Explicit owner action;
    declines politely when KINECT_GUARD_ENABLED is off (arming isn't allowed)."""
    if not _cfg_flag("KINECT_GUARD_ENABLED"):
        return ("Guard mode is switched off in settings, sir — enable "
                "KINECT_GUARD_ENABLED first and I'll be able to stand watch.")
    with _guard_lock:
        already = _armed[0]
    if already:
        return "I'm already standing watch, sir."

    _reset_session()
    with _guard_lock:
        _armed[0] = True
        _armed_since[0] = time.time()
        # Bump the arm generation and ALWAYS start a fresh thread for it. A
        # prior session's thread may still be alive mid-wind-down — its stale
        # generation makes it exit on its next tick, so there's no aliveness
        # check to race against (the old TOCTOU could leave guard armed with
        # no poller at all).
        _monitor_gen[0] += 1
        gen = _monitor_gen[0]
    t = threading.Thread(target=_monitor_loop, args=(gen,), daemon=True,
                         name="guard-mode-monitor")
    t.start()
    _monitor_thread[0] = t

    # Name the cameras we'll be watching so the confirmation is concrete.
    n = _available_camera_count()
    cams = (f" I'll be watching {n} "
            + ("camera" if n == 1 else "cameras") + ".") if n else ""
    return f"Standing watch, sir — I'll alert you to any movement.{cams}"


def guard_off(_: str = "") -> str:
    """Disarm + stop the monitor daemon."""
    with _guard_lock:
        if not _armed[0]:
            return "I wasn't on watch, sir."
        _armed[0] = False
        _armed_since[0] = 0.0
        events = _event_count[0]
    # The loop sees armed=False on its next tick and returns; nudge the handle.
    _monitor_thread[0] = None
    if events:
        return (f"Standing down, sir. I logged {events} "
                + ("movement" if events == 1 else "movements")
                + " while on watch.")
    return "Standing down, sir. All quiet while I watched."


def guard_status(_: str = "") -> str:
    """Armed/disarmed, for how long, and how many events so far."""
    with _guard_lock:
        armed = _armed[0]
        since = _armed_since[0]
        events = _event_count[0]
        last = dict(_last_event[0]) if _last_event[0] else None

    if not armed:
        if not _cfg_flag("KINECT_GUARD_ENABLED"):
            return ("I'm not on watch, sir — guard mode is disabled in settings "
                    "(enable KINECT_GUARD_ENABLED to use it).")
        return "I'm not currently on watch, sir. Say 'guard the room' to arm me."

    now = time.time()
    dur = _format_seconds(max(0.0, now - since)) if since else "a moment"
    ev_txt = ("no movement yet" if events == 0
              else f"{events} " + ("movement" if events == 1 else "movements"))
    line = f"On watch for {dur}, sir — {ev_txt}."
    if last and events:
        ago = _format_seconds(max(0.0, now - last.get("ts", now)))
        cam = last.get("camera", "a camera")
        line += f" Last was on {cam}, {ago} ago."
    return line


def _available_camera_count() -> int:
    """How many sources are likely to be watched right now (webcams with a
    cached frame + the Kinect if streaming). Best-effort, for the arm message."""
    try:
        return len(_collect_frames())
    except Exception:
        return 0


def _format_seconds(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f"{s} second{'s' if s != 1 else ''}"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    h, m = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["guard_on"]     = guard_on
    actions["guard_off"]    = guard_off
    actions["guard_status"] = guard_status
    print("  [guard] guard-mode actions registered "
          "(guard_on, guard_off, guard_status) — armed on explicit request only")
