"""
Face / gaze tracker skill for JARVIS.

The existing _face_tracking_thread in bobert_companion.py already runs OpenCV
face detection on every configured webcam and updates two shared dicts:
  bobert_companion._camera_last_seen     — index -> last-detected timestamp
  bobert_companion._camera_latest_frame  — index -> most recent BGR frame
guarded by bobert_companion._camera_state_lock.

This skill builds a higher-level "gaze" layer on top of that raw signal:

  • Background poller (every GAZE_POLL_INTERVAL) reads camera last-seen times,
    classifies the user's current side (left-only / right-only / both / none),
    maps it to a monitor name via the existing MONITORS layout, and applies
    hysteresis (HYSTERESIS_SAMPLES consecutive identical reads required) so a
    half-second glance doesn't flip the reported state.

  • Maintains per-monitor dwell totals + longest single dwell + total
    face-visible time since the skill loaded.

  • Registers three new actions:
      gaze_status        — "currently looking at the right monitor, sir"
      gaze_stats         — dwell totals + most-watched monitor
      face_track_status  — alias for gaze_status

  • Wraps the existing which_monitor action with a fast-path: if the cached
    gaze state is fresh (< GAZE_CACHE_FRESH s old) and unambiguous
    (left-only / right-only / not-visible) we answer from the cache and skip
    the recompute. The "both cameras see the face" case still falls through
    to the original action because it needs vision to disambiguate top vs.
    middle.

  • Wraps see_user to append a one-line note about the current gaze target.
"""
import importlib
import sys
import threading
import time


# ─── tunables ────────────────────────────────────────────────────────────
GAZE_POLL_INTERVAL  = 0.5    # how often the poller wakes
# Bumped 2.0 → 3.0 (2026-05-29 task 11:26): user reported gaze loss while
# clearly in frame. Wider window tolerates the 1-2 s gaps the cascade hits
# between detections without flipping the reported state to "away".
FACE_FRESH_SECONDS  = 3.0    # camera "sees user now" window
HYSTERESIS_SAMPLES  = 2      # consecutive identical reads before commit
GAZE_CACHE_FRESH    = 3.0    # fast-path only if last sample < this old
INITIAL_DELAY_SECONDS = 5    # let the face-track thread come up first

# ─── Kinect presence (opt-in; all gated by core.config flags) ────────────
# When KINECT_PRESENCE_ENABLED, the poller merges true skeleton presence from
# the Kinect v2 into the gaze state. When KINECT_PRESENCE_STANDBY, a sustained
# empty room drops JARVIS to standby; when KINECT_PRESENCE_WAKE, a person
# reappearing clears standby. Hysteresis windows keep a one-frame body-track
# dropout from flapping the state.
KINECT_EMPTY_STANDBY_SECONDS = 180.0   # room empty this long → standby
KINECT_PRESENCE_REREAD_SEC   = 0.0     # presence is read every poll tick
# ─────────────────────────────────────────────────────────────────────────


_state_lock = threading.Lock()
_state = {
    "current_monitor": None,      # "left" / "right" / "middle_or_top" / "away" / None
    "current_sides":   None,      # frozenset of sides currently visible
    "last_sample_at":  0.0,
    "monitor_since":   0.0,       # when did current_monitor become current
    "face_visible":    False,
    "last_face_at":    0.0,
    "first_face_at":   0.0,       # first ever face sighting since skill loaded
    # Kinect skeleton presence (populated only when KINECT_PRESENCE_ENABLED):
    "kinect_present":  None,      # bool | None (None = no Kinect reading yet)
    "kinect_count":    0,         # tracked-body count
    "kinect_nearest_m": None,     # nearest body distance (metres)
    "kinect_facing":   None,      # any body facing the sensor
    "kinect_at":       0.0,       # monotonic ts of last Kinect presence read
    "kinect_last_present_at": 0.0,   # last time a body WAS present
}

# Empty-room → standby bookkeeping. _kinect_empty_since records when the room
# first went empty (0.0 = currently occupied / unknown); the standby fires only
# after it's held empty for KINECT_EMPTY_STANDBY_SECONDS. List-wrapped so the
# poller can mutate without `global`.
_kinect_empty_since = [0.0]
_kinect_standby_fired = [False]   # latch: don't re-engage every tick

# Dwell tracking: name -> total seconds, name -> longest single run
_dwell_total: dict[str, float] = {}
_dwell_longest: dict[str, float] = {}
_face_visible_total = [0.0]      # list-wrapped so the loop can mutate it

# Hysteresis buffer — last few raw observations
_pending_monitor: list[str | None] = []


def _classify_sides(bc) -> tuple[frozenset[str], dict[int, str]]:
    """Read camera last-seen times and return (set of sides visible, side map).

    side ∈ {"left", "right"} based on each camera's look_x preset (matches
    the heuristic _act_which_monitor already uses)."""
    cameras = getattr(bc, "CAMERAS", []) or []
    side_for_idx = {
        cam["index"]: ("left" if cam.get("look_x", 0.5) < 0.5 else "right")
        for cam in cameras
    }
    now = time.time()
    seen_dict = getattr(bc, "_camera_last_seen", {}) or {}
    lock = getattr(bc, "_camera_state_lock", None)
    sides: set[str] = set()
    if lock is not None:
        with lock:
            for idx, ts in seen_dict.items():
                if ts and (now - ts) <= FACE_FRESH_SECONDS:
                    s = side_for_idx.get(idx)
                    if s:
                        sides.add(s)
    else:
        for idx, ts in seen_dict.items():
            if ts and (now - ts) <= FACE_FRESH_SECONDS:
                s = side_for_idx.get(idx)
                if s:
                    sides.add(s)
    return frozenset(sides), side_for_idx


def _monitor_name_from_sides(bc, sides: frozenset[str]) -> str:
    """Translate the visible-sides set into a monitor label.

    For the "both sides" case we return "middle_or_top" — disambiguating
    those two requires a vision call which is intentionally NOT done in the
    fast poller (we don't want to ping Claude every 500ms). The on-demand
    which_monitor action still handles that disambiguation when asked.
    """
    monitors = getattr(bc, "MONITORS", {}) or {}
    if not sides:
        return "away"
    if sides == frozenset({"left"}):
        return "left"
    if sides == frozenset({"right"}):
        return "right"
    # both sides — could be the middle or the top monitor
    has_top = "top" in monitors
    has_mid = "middle" in monitors
    if has_top and has_mid:
        return "middle_or_top"
    if has_top:
        return "top"
    return "middle"


def _commit_state(new_monitor: str, sides: frozenset[str], now: float) -> None:
    """Update _state + dwell stats when monitor changes.
    Called under _state_lock."""
    prev = _state["current_monitor"]
    if prev == new_monitor:
        _state["last_sample_at"] = now
        _state["current_sides"]  = sides
        return

    # Close out the previous run
    if prev is not None and _state["monitor_since"]:
        run = max(0.0, now - _state["monitor_since"])
        if prev != "away":
            _dwell_total[prev]   = _dwell_total.get(prev, 0.0) + run
            if run > _dwell_longest.get(prev, 0.0):
                _dwell_longest[prev] = run

    _state["current_monitor"] = new_monitor
    _state["current_sides"]   = sides
    _state["monitor_since"]   = now
    _state["last_sample_at"]  = now


# ─── Kinect presence integration (all opt-in via core.config flags) ──────

def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence (early
    boot / standalone test). Read fresh each call so a Settings toggle takes
    effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _kinect_bridge():
    """Return the live kinect_bridge module, or None. Prefer the already-loaded
    instance (bobert_companion imports it as audio.kinect_bridge); fall back to
    a direct import so the skill works even if the monolith hasn't loaded it."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _read_kinect_presence() -> dict | None:
    """Fetch get_presence() from the bridge, or None if Kinect presence is
    disabled / the bridge is unavailable / it isn't actually streaming.
    NEVER raises."""
    if not _cfg_flag("KINECT_PRESENCE_ENABLED"):
        return None
    kb = _kinect_bridge()
    if kb is None:
        return None
    try:
        ok, _reason = kb.available()
        if not ok:
            return None
        return kb.get_presence()
    except Exception:
        return None


def _merge_kinect_presence(presence: dict, now: float) -> None:
    """Fold a Kinect get_presence() reading into _state. Caller holds
    _state_lock."""
    present = bool(presence.get("present"))
    _state["kinect_present"]   = present
    _state["kinect_count"]     = int(presence.get("count", 0) or 0)
    _state["kinect_nearest_m"] = presence.get("nearest_m")
    _state["kinect_facing"]    = presence.get("facing")
    _state["kinect_at"]        = now
    if present:
        _state["kinect_last_present_at"] = now
        # A real skeleton beats the Haar guess: count it as a face sighting so
        # gaze_status/see_user don't report "not in view" while the Kinect
        # clearly sees a body.
        if not _state["first_face_at"]:
            _state["first_face_at"] = now
        _state["last_face_at"] = now


def _bc():
    """The live monolith module (main or by-name), or None. Mirrors the
    standby_audio_detect lookup pattern."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _apply_kinect_presence_actions(present: bool, now: float) -> None:
    """Drive wake/standby from Kinect presence — ONLY behind the opt-in flags.
    Empty room for a sustained window → _standby_auto_engage('room_empty');
    a person reappearing while in standby → clear standby (force_wake path).
    Swallows every error so a presence read never crashes the poller."""
    want_standby = _cfg_flag("KINECT_PRESENCE_STANDBY")
    want_wake    = _cfg_flag("KINECT_PRESENCE_WAKE")
    if not (want_standby or want_wake):
        _kinect_empty_since[0] = 0.0
        return
    bc = _bc()
    if bc is None:
        return

    if present:
        # Room occupied: reset the empty timer + standby latch, and wake if
        # we're dormant and the user opted into wake-on-presence.
        _kinect_empty_since[0] = 0.0
        _kinect_standby_fired[0] = False
        if want_wake:
            try:
                in_standby = bool(getattr(bc, "_standby_mode")[0]) or \
                             bool(getattr(bc, "_sleep_mode")[0])
            except Exception:
                in_standby = False
            if in_standby:
                _kinect_clear_standby(bc)
        return

    # Room empty.
    if not want_standby:
        return
    if _kinect_empty_since[0] == 0.0:
        _kinect_empty_since[0] = now
        return
    if _kinect_standby_fired[0]:
        return
    if (now - _kinect_empty_since[0]) < KINECT_EMPTY_STANDBY_SECONDS:
        return
    # Don't re-engage if already dormant.
    try:
        if bool(getattr(bc, "_standby_mode")[0]) or bool(getattr(bc, "_sleep_mode")[0]):
            _kinect_standby_fired[0] = True
            return
    except Exception:
        pass
    engage = getattr(bc, "_standby_auto_engage", None)
    if not callable(engage):
        return
    try:
        if engage("room_empty"):
            print("  [face-track] Kinect saw an empty room for "
                  f"{KINECT_EMPTY_STANDBY_SECONDS:.0f}s — entering standby")
    except Exception as e:
        print(f"  [face-track] standby-on-empty failed: {e}")
    _kinect_standby_fired[0] = True


def _kinect_clear_standby(bc) -> None:
    """Clear sleep/standby the way the force_wake tray path does — under the
    same lock so a concurrent auto-engage can't re-assert it. Best-effort."""
    lock = getattr(bc, "_standby_auto_engage_lock", None)
    try:
        sleep_flag = getattr(bc, "_sleep_mode", None)
        standby_flag = getattr(bc, "_standby_mode", None)
        if sleep_flag is None or standby_flag is None:
            return
        if lock is not None:
            with lock:
                sleep_flag[0] = False
                standby_flag[0] = False
        else:
            sleep_flag[0] = False
            standby_flag[0] = False
        try:
            bc._write_hud_state(sleep_mode=False, standby_mode=False, state="Idle")
        except Exception:
            pass
        print("  [face-track] Kinect saw someone return — cleared standby")
    except Exception as e:
        print(f"  [face-track] wake-on-presence failed: {e}")


def _poll_once(bc) -> None:
    sides, _side_map = _classify_sides(bc)
    raw_monitor = _monitor_name_from_sides(bc, sides)
    now = time.time()

    # Kinect skeleton presence (opt-in). Read it once per tick and fold it into
    # the gaze state; then optionally drive wake/standby behind its own flags.
    kinect = _read_kinect_presence()
    if kinect is not None:
        with _state_lock:
            _merge_kinect_presence(kinect, now)
        try:
            _apply_kinect_presence_actions(bool(kinect.get("present")), now)
        except Exception as e:   # pragma: no cover - defensive: never crash the poller
            print(f"  [face-track] kinect presence-action error: {e}")

    # Hysteresis: only commit when the same reading has held for N samples
    _pending_monitor.append(raw_monitor)
    if len(_pending_monitor) > HYSTERESIS_SAMPLES:
        del _pending_monitor[: len(_pending_monitor) - HYSTERESIS_SAMPLES]

    stable = (
        len(_pending_monitor) >= HYSTERESIS_SAMPLES
        and all(m == _pending_monitor[0] for m in _pending_monitor)
    )

    with _state_lock:
        # Face-visible accounting — independent of hysteresis (uses raw signal)
        was_visible = _state["face_visible"]
        is_visible  = bool(sides)
        if is_visible:
            if not _state["first_face_at"]:
                _state["first_face_at"] = now
            _state["last_face_at"] = now
            if was_visible and _state["last_sample_at"]:
                _face_visible_total[0] += max(0.0, now - _state["last_sample_at"])
        _state["face_visible"] = is_visible

        if stable:
            _commit_state(_pending_monitor[0], sides, now)
        else:
            _state["last_sample_at"] = now
            _state["current_sides"]  = sides


def _poll_loop() -> None:  # pragma: no cover - non-terminating background daemon (sleeps INITIAL_DELAY then polls forever); each tick delegates to _poll_once, which is unit-tested directly
    time.sleep(INITIAL_DELAY_SECONDS)
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception as e:
        print(f"  [face-track] could not import bobert_companion: {e}")
        return
    while True:
        try:
            _poll_once(bc)
        except Exception as e:
            print(f"  [face-track] poll error: {e}")
        time.sleep(GAZE_POLL_INTERVAL)


# ─── helpers used by actions ─────────────────────────────────────────────

def _format_seconds(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f"{s} second{'s' if s != 1 else ''}"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    h, m = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"


def _monitor_phrase(name: str | None) -> str:
    if name is None:
        return "not yet established"
    if name == "away":
        return "no monitor — user not visible"
    if name == "middle_or_top":
        return "the middle or top monitor"
    return f"the {name} monitor"


def _snapshot_state() -> dict:
    with _state_lock:
        return dict(_state)


# ─── read-failure spike export (consumed by skills/self_diagnostic) ──────
# The raw consecutive-fails counter lives in bobert_companion._face_tracking_
# thread (entry["fails"]) and is published via _note_camera_read_attempt into
# get_camera_failure_summary(). We expose a stable per-skill name so the
# self-diagnostic auto-queue probe doesn't have to reach into bobert internals
# directly.
_FACE_READ_FAILURE_SPIKE_THRESHOLD = 5


def get_consecutive_read_failures() -> dict[int, dict]:
    """Per-camera-index snapshot of the current consecutive-fail count and
    related metadata. Returns ``{}`` when bobert_companion isn't loaded
    (early boot, standalone test). Each value contains:
        consecutive_fails, max_consecutive_fails, last_error, last_error_at,
        last_ok_at, total_fails.
    """
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return {}
    fn = getattr(bc, "get_camera_failure_summary", None)
    if not callable(fn):
        return {}
    try:
        return fn() or {}
    except Exception:
        return {}


def get_read_failure_spike_signals(
    threshold: int = _FACE_READ_FAILURE_SPIKE_THRESHOLD,
) -> list[dict]:
    """Return one signal per camera whose consecutive-read-failures crossed
    the spike threshold. Used by skills/self_diagnostic to decide whether to
    auto-queue a face_tracker fix request. Each entry has:
        cam_index, consecutive_fails, max_consecutive_fails, last_error,
        seconds_since_last_ok.
    """
    out: list[dict] = []
    now = time.time()
    for idx, info in (get_consecutive_read_failures() or {}).items():
        try:
            consec = int(info.get("consecutive_fails", 0))
            peak   = int(info.get("max_consecutive_fails", 0))
        except Exception:
            continue
        # Either CURRENT consec >= threshold (live spike) or the historical
        # peak >= threshold while the camera still hasn't recovered (last_ok_at
        # never updated). Both deserve a fix request.
        last_ok = float(info.get("last_ok_at", 0.0) or 0.0)
        gap_s = (now - last_ok) if last_ok else float("inf")
        live_spike      = consec >= threshold
        historic_spike  = peak >= threshold and gap_s > 30.0
        if live_spike or historic_spike:
            out.append({
                "cam_index":               idx,
                "consecutive_fails":       consec,
                "max_consecutive_fails":   peak,
                "last_error":              info.get("last_error"),
                "seconds_since_last_ok":   round(gap_s, 1) if gap_s != float("inf") else None,
            })
    return out


# ─── actions ─────────────────────────────────────────────────────────────

def _kinect_presence_note(snap: dict, now: float) -> str:
    """A short spoken clause about Kinect skeleton presence when it's fresh,
    else ''. Used to enrich gaze_status with the stronger signal."""
    if snap.get("kinect_present") is None or not snap.get("kinect_at"):
        return ""
    # kinect_at is monotonic; compare against a monotonic now.
    if (time.monotonic() - snap["kinect_at"]) > 5.0:
        return ""
    count = snap.get("kinect_count", 0)
    if not snap.get("kinect_present") or count <= 0:
        return ""
    nearest = snap.get("kinect_nearest_m")
    who = "one person" if count == 1 else f"{count} people"
    if nearest:
        return f" The Kinect sees {who} about {nearest:.1f} metres away."
    return f" The Kinect sees {who} in the room."


def gaze_status(_: str = "") -> str:
    snap = _snapshot_state()
    if not snap["last_sample_at"]:
        return "Face tracker is still warming up, sir."

    now = time.time()
    monitor = snap["current_monitor"]
    kinect_note = _kinect_presence_note(snap, now)
    if monitor is None:
        if kinect_note:
            return f"I haven't pinned your gaze yet, sir, but{kinect_note}"
        return "I haven't established your gaze yet, sir."

    if monitor == "away":
        # The Kinect skeleton is a stronger presence signal than the Haar
        # cascade — if it sees a body, say so rather than "not in view".
        if kinect_note:
            return f"You're off-camera for the webcams, sir, but{kinect_note}"
        if snap["last_face_at"]:
            ago = _format_seconds(now - snap["last_face_at"])
            return f"You're not currently in view, sir — last seen {ago} ago."
        return "I haven't seen you at all this session, sir."

    dwell = _format_seconds(now - snap["monitor_since"])
    where = _monitor_phrase(monitor)
    return f"You're looking at {where}, sir — for the past {dwell}."


def gaze_stats(_: str = "") -> str:
    snap = _snapshot_state()
    now = time.time()

    # Close out the current run so totals reflect "right now"
    totals = dict(_dwell_total)
    if snap["current_monitor"] and snap["current_monitor"] != "away" and snap["monitor_since"]:
        run = max(0.0, now - snap["monitor_since"])
        totals[snap["current_monitor"]] = totals.get(snap["current_monitor"], 0.0) + run

    if not totals:
        return "I have no gaze history yet, sir."

    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    top_name, top_secs = ranked[0]
    parts = [f"{name} {_format_seconds(secs)}" for name, secs in ranked]
    face_seen = _face_visible_total[0]
    extra = ""
    if face_seen > 0:
        extra = f" I've had you in view for roughly {_format_seconds(face_seen)} this session."
    return (
        f"Most of your attention has gone to the {top_name} monitor, sir — "
        f"breakdown: {', '.join(parts)}.{extra}"
    )


def face_track_status(_: str = "") -> str:
    return gaze_status(_)


# ─── wrappers around existing actions ────────────────────────────────────

def _build_which_monitor_wrapper(original):
    """Fast-path: if the cached gaze state is fresh and unambiguous, answer
    from the cache. Fall through to the original action otherwise."""
    def wrapper(arg: str = "") -> str:
        snap = _snapshot_state()
        now  = time.time()
        fresh = snap["last_sample_at"] and (now - snap["last_sample_at"]) < GAZE_CACHE_FRESH
        monitor = snap["current_monitor"]
        if fresh and monitor in ("left", "right"):
            # Match _act_which_monitor: only append "(name)" when the side is
            # actually present in MONITORS.
            try:
                bc = importlib.import_module("bobert_companion")
                monitors = getattr(bc, "MONITORS", {}) or {}
            except Exception:
                monitors = {}
            suffix = f" ({monitor})" if monitor in monitors else ""
            return f"facing {monitor.upper()} monitor{suffix}"
        if fresh and monitor == "away":
            return "user is not visible to any camera — can't determine monitor"
        # "middle_or_top" or stale/None → delegate to the original (which can
        # call vision to disambiguate top vs. middle).
        return original(arg)
    return wrapper


def _build_see_user_wrapper(original):
    """Append a single-line gaze note to the existing see_user description."""
    def wrapper(arg: str = "") -> str:
        result = original(arg)
        snap = _snapshot_state()
        monitor = snap["current_monitor"]
        if monitor and monitor != "away":
            note = f"\n\n[gaze: currently looking at {_monitor_phrase(monitor)}]"
            return f"{result}{note}"
        if monitor == "away":
            return f"{result}\n\n[gaze: user not currently in view]"
        return result
    return wrapper


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["gaze_status"]       = gaze_status
    actions["gaze_stats"]        = gaze_stats
    actions["face_track_status"] = face_track_status

    if "which_monitor" in actions:
        # INTENTIONAL_WRAP: fast-path cached gaze state (jarvis_todo task #28).
        actions["which_monitor"] = _build_which_monitor_wrapper(actions["which_monitor"])
    if "see_user" in actions:
        # INTENTIONAL_WRAP: same — cached gaze fast-path.
        actions["see_user"] = _build_see_user_wrapper(actions["see_user"])

    # Guard against duplicate pollers on skill reload: load_skills() re-execs
    # this module (fresh globals), so a module flag can't see a prior load's
    # still-running thread — only an OS-thread name check survives.
    if any(th.name == "face-tracker-skill" and th.is_alive()
           for th in threading.enumerate()):
        print("  [face-track] poller already running — skipping duplicate "
              "(skill reload)")
    else:
        t = threading.Thread(target=_poll_loop, daemon=True,
                             name="face-tracker-skill")
        t.start()
        print(
            f"  [face-track] gaze poller active "
            f"(every {GAZE_POLL_INTERVAL:.1f}s, hysteresis {HYSTERESIS_SAMPLES} samples)"
        )
