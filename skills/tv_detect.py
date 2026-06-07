"""
tv_detect skill — the LIVE wiring for the camera-based "is the TV on?" detector.

THE FEATURE
===========
The owner watches TV. JARVIS already suppresses ingesting TV chatter as the
owner's facts via AUDIO (SMTC now-playing + the spectral music detector + a
local-LLM content judge + voice-ID). This skill adds an INDEPENDENT VISUAL veto:
if a camera sees a bright, flickering screen (a powered-on TV/monitor), it
contributes one more OR-signal to ambient-learning suppression — so a TV the
audio gates miss (muted, an unrecognised stream, a show the judge can't place)
still stops ambient learning.

The pure image statistics + the rolling hysteresis decision + the calibration
store live in audio/tv_detect.py (numpy-only, NO cv2, so they run on CI). THIS
file is the live wiring around them:

  • reads ONE frame from the monolith's existing face-tracker cache
    (_camera_latest_frame, under _camera_state_lock) — it does NOT open a camera
    and does NOT touch the face-track loop;
  • keeps a rolling window of (brightness, temporal-variance) readings, sampled
    no faster than POLL_MIN_INTERVAL_S, and feeds them to a TVDecider;
  • exposes is_tv_on() — the public predicate _ambient_media_is_playing() OR's in;
  • voice actions to calibrate the region, report status, and toggle the flag.

EVERYTHING is opt-in + safe (mirrors skills/kinect_pointing.py):
  • Gated by core.config.TV_DETECT_ENABLED (default False), re-read each call so a
    Settings/voice toggle takes effect with no restart. With it off, is_tv_on()
    is always False and NO frame is ever read.
  • PURELY a suppression signal: it can only veto ambient learning, never trigger
    an action — so it's safe in staging too (no special staging gate needed; it
    drives nothing). is_tv_on() simply returns False unless explicitly enabled.
  • The calibration rectangle is a SEPARATE gitignored json
    (data/tv_region.json via JARVIS_TV_REGION_PATH) with an atomic write — it
    never touches user_settings.json. The on/off flag persists via the SAME
    hardened Settings writer model_picker / kinect_pointing use.
  • All frame contact degrades to "no reading" (is_tv_on() False) — a missing
    camera, numpy, or detector module never raises into the ambient seam.

Voice actions:
  calibrate_tv_region            — store the TV's rectangle in the frame.
                                   "calibrate the tv region".
  tv_detect_status               — is detection on, calibrated, and does it see a
                                   TV right now. "tv detection status".
  tv_detect_on / tv_detect_off   — toggle TV_DETECT_ENABLED live + persisted.
                                   "turn on/off tv detection".
"""
from __future__ import annotations

import sys
import threading
import time


# ─── tunables ────────────────────────────────────────────────────────────
# Don't sample the cached frame faster than this — the detector only needs a
# reading every second-ish to build sustained evidence, and this keeps the cost
# (one luma mean + one abs-diff over a small region) negligible. is_tv_on() is
# called from the ambient gate which itself fires per overheard utterance, so we
# rate-limit the actual pixel work here rather than recompute every call.
POLL_MIN_INTERVAL_S = 1.0
# Calibration: sample a few frames to confirm the camera is actually feeding
# (we don't auto-find the TV rectangle — the owner frames it; v1 stores the WHOLE
# current frame as the region, which is the safe default, and the rectangle is
# tunable on disk). Kept tiny so calibrate returns promptly.
CALIBRATE_CONFIRM_FRAMES = 3
CALIBRATE_POLL_INTERVAL = 0.1


# ─── module seams (lazy; never raise at import) ────────────────────────────
def _detect_module():
    """The pure stats + store module (audio.tv_detect). Imported lazily so a
    failure can't stop the skill registering its voice actions."""
    mod = sys.modules.get("audio.tv_detect")
    if mod is not None:
        return mod
    try:
        from audio import tv_detect as _td
        return _td
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings/voice toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _detect_enabled() -> bool:
    """The master gate. PURELY a suppression signal, so unlike the Kinect skills
    there is no staging carve-out — it never drives a device, it only vetoes
    learning, which is the SAME behaviour we want in staging."""
    return _cfg_flag("TV_DETECT_ENABLED")


# ─── calibration store accessor ────────────────────────────────────────────
def _store():
    """A fresh TVRegionStore bound to the configured (gitignored) json path, or
    None if the pure module is unavailable. Cheap to recreate — it reads the file
    per call so a concurrent calibrate is always seen."""
    td = _detect_module()
    if td is None:
        return None
    try:
        return td.TVRegionStore()
    except Exception:
        return None


# ─── live decider (module-singleton rolling window) ────────────────────────
_decider_lock = threading.Lock()
_decider = None              # audio.tv_detect.TVDecider, lazily built
_prev_frame = [None]         # list-box: the previous sampled frame (for delta)
_last_poll = [0.0]           # list-box: monotonic ts of the last pixel sample


def _get_decider():
    """The module-level TVDecider, built lazily from the pure module. None if the
    detector module isn't importable."""
    global _decider
    if _decider is not None:
        return _decider
    td = _detect_module()
    if td is None:
        return None
    try:
        _decider = td.TVDecider()
    except Exception:
        _decider = None
    return _decider


def _snapshot_primary_frame():
    """Return a COPY of the most-recent cached camera frame and its age in
    seconds — (frame, age_s) — or (None, None) when no fresh frame is available.

    Reads the monolith's face-tracker cache (_camera_latest_frame /
    _camera_last_frame_at) UNDER _camera_state_lock, exactly like core.actions'
    see_user path. Prefers the camera that most recently saw a face (the one
    pointed at the user → most likely to also frame the TV), else any cached
    frame. Never raises — any problem reads as (None, None)."""
    bc = _bc()
    if bc is None:
        return None, None
    lock = getattr(bc, "_camera_state_lock", None)
    frames = getattr(bc, "_camera_latest_frame", None)
    if lock is None or not isinstance(frames, dict):
        return None, None
    try:
        last_seen = getattr(bc, "_camera_last_seen", {}) or {}
        frame_at = getattr(bc, "_camera_last_frame_at", {}) or {}
        with lock:
            if not frames:
                return None, None
            # Prefer the most-recently-face-seen camera; else any cached index.
            best_idx = None
            best_seen = -1.0
            for idx in frames.keys():
                seen = float(last_seen.get(idx, 0.0))
                if seen > best_seen:
                    best_seen = seen
                    best_idx = idx
            if best_idx is None:
                best_idx = next(iter(frames))
            frame = frames.get(best_idx)
            if frame is None:
                return None, None
            try:
                frame = frame.copy()
            except Exception:
                pass
            ts = float(frame_at.get(best_idx, 0.0))
    except Exception:
        return None, None
    age = (time.time() - ts) if ts else None
    return frame, age


def _poll_once(now_fn=None) -> None:
    """Sample the cached frame at most once per POLL_MIN_INTERVAL_S, compute its
    brightness + temporal delta vs the previous sample over the calibrated
    region, and feed the reading to the decider. Best-effort + silent — any error
    leaves the decider untouched (a gap reads as a non-qualifying frame as it
    ages out). Holds _decider_lock for the (cheap) update. `now_fn` defaults to
    ``time.monotonic`` resolved AT CALL TIME (not bound as a default) so a test
    can patch the module clock and have it honoured here."""
    td = _detect_module()
    decider = _get_decider()
    if td is None or decider is None:
        return
    now = (now_fn or time.monotonic)()
    with _decider_lock:
        if (now - _last_poll[0]) < POLL_MIN_INTERVAL_S:
            return
        _last_poll[0] = now
        frame, age = _snapshot_primary_frame()
        # No frame, or a stale one (camera stopped feeding) → record a
        # non-qualifying reading so a frozen last-frame can't latch the verdict.
        if frame is None or (age is not None and age > td.READING_MAX_AGE_S):
            decider.observe(None, None)
            _prev_frame[0] = None
            return
        region = None
        store = _store()
        if store is not None:
            try:
                region = store.get_region()
            except Exception:
                region = None
        brightness = td.frame_brightness(frame, region)
        prev = _prev_frame[0]
        delta = td.frame_temporal_delta(prev, frame, region) if prev is not None else None
        decider.observe(brightness, delta)
        _prev_frame[0] = frame


def is_tv_on() -> bool:
    """PUBLIC PREDICATE — the OR-signal _ambient_media_is_playing() consults.

    True iff TV detection is ENABLED and the rolling visual evidence currently
    says a bright, flickering screen is ON. Polls the cached frame (rate-limited)
    as a side effect so the verdict stays live even though it's only asked
    per-utterance. FAIL-SAFE: returns False on ANY problem (disabled, no detector
    module, no camera frame, numpy missing) so a probe glitch can NEVER suppress
    learning on its own — it can only ADD a veto when it positively sees a TV.
    Never raises."""
    try:
        if not _detect_enabled():
            return False
        td = _detect_module()
        decider = _get_decider()
        if td is None or decider is None:
            return False
        _poll_once()
        return bool(decider.is_on(now=time.time()))
    except Exception:
        return False


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
    other settings — the EXACT path model_picker / kinect_pointing use
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
    """Flip TV_DETECT_ENABLED live (core.config) and persist it."""
    try:
        import core.config as _cfg
        _cfg.TV_DETECT_ENABLED = bool(on)
    except Exception:
        pass
    return _persist_setting("TV_DETECT_ENABLED", bool(on))


# ─── actions ─────────────────────────────────────────────────────────────
def calibrate_tv_region(_: str = "") -> str:
    """Calibrate the region of the camera frame the TV occupies. The owner frames
    the camera so the TV fills it and says 'calibrate the tv region'. v1 stores
    the WHOLE current frame as the region (the safe default — the detector then
    watches the entire view); the stored rectangle is normalised and tunable on
    disk (data/tv_region.json) for a tighter crop later. Honest on every failure
    — never claims to have calibrated when no camera frame is available."""
    if not _detect_enabled():
        return ("TV detection is off, sir — say 'turn on tv detection' first, "
                "then frame the TV in a camera and calibrate it.")
    td = _detect_module()
    if td is None:
        return "My TV-detector module didn't load, sir — I can't calibrate."
    store = _store()
    if store is None:
        return "My TV-region store didn't load, sir — I can't save that."

    # Confirm the camera is actually feeding frames before we claim a region.
    saw_frame = False
    for _i in range(max(1, CALIBRATE_CONFIRM_FRAMES)):
        frame, age = _snapshot_primary_frame()
        if frame is not None and (age is None or age <= td.READING_MAX_AGE_S):
            saw_frame = True
            break
        time.sleep(CALIBRATE_POLL_INTERVAL)
    if not saw_frame:
        return ("I can't see a camera frame right now, sir — make sure a camera "
                "is on and pointed at the TV, then calibrate again.")

    # Store the whole frame as the watched region (x=0,y=0,w=1,h=1).
    if not store.put_region(0.0, 0.0, 1.0, 1.0):
        return ("I saw the camera, sir, but couldn't save the TV region to disk.")
    # A fresh region invalidates the rolling history (it was measuring the old
    # crop) — reset so the next readings build evidence on the new region.
    with _decider_lock:
        d = _get_decider()
        if d is not None:
            d.reset()
        _prev_frame[0] = None
        _last_poll[0] = 0.0
    return ("Calibrated, sir — I'll watch the whole camera view for a lit, "
            "moving screen and use it to avoid learning from the TV. You can "
            "tighten the region in data/tv_region.json if you'd like.")


def tv_detect_status(_: str = "") -> str:
    """Report whether TV detection is on, whether a region is calibrated, and
    whether it sees a TV ON right now. 'tv detection status'."""
    enabled = _detect_enabled()
    store = _store()
    calibrated = bool(store and store.is_calibrated())
    region_note = ("a calibrated region" if calibrated
                   else "the whole frame (uncalibrated)")
    if not enabled:
        return (f"TV detection is off, sir — say 'turn on tv detection' to enable "
                f"it. It would watch {region_note}.")
    td = _detect_module()
    if td is None:
        return ("TV detection is on, sir, but my detector module didn't load, so "
                "I can't actually watch the screen right now.")
    # Take a live reading so the status reflects NOW.
    try:
        _poll_once()
    except Exception:
        pass
    frame, age = _snapshot_primary_frame()
    if frame is None:
        return (f"TV detection is on and watching {region_note}, sir, but I'm not "
                "getting any camera frames at the moment.")
    seeing = is_tv_on()
    if seeing:
        return (f"TV detection is on, sir — and right now I DO see a lit, moving "
                f"screen ({region_note}), so I'll hold off learning from the room "
                "audio.")
    return (f"TV detection is on and watching {region_note}, sir — and right now I "
            "do not see a TV on.")


def tv_detect_on(_: str = "") -> str:
    """Turn TV detection on (live + persisted)."""
    already = _detect_enabled()
    persisted = _set_enabled(True)
    store = _store()
    calibrated = bool(store and store.is_calibrated())
    cal_note = ("" if calibrated else
                " It's watching the whole frame — say 'calibrate the tv region' "
                "while a camera frames the TV for a tighter focus.")
    if already:
        return "TV detection is already on, sir." + cal_note
    msg = ("TV detection on, sir — I'll use a lit, flickering screen as another "
           "reason not to learn from the TV.")
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg + cal_note


def tv_detect_off(_: str = "") -> str:
    """Turn TV detection off (live + persisted)."""
    if not _detect_enabled():
        return "TV detection is already off, sir."
    persisted = _set_enabled(False)
    # Clear the rolling evidence so a later re-enable starts fresh.
    with _decider_lock:
        d = _get_decider()
        if d is not None:
            d.reset()
        _prev_frame[0] = None
    msg = "TV detection off, sir."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


# ─── registration ────────────────────────────────────────────────────────
def register(actions):
    actions["calibrate_tv_region"] = calibrate_tv_region
    actions["tv_calibrate"]        = calibrate_tv_region
    actions["tv_detect_status"]    = tv_detect_status
    actions["tv_status"]           = tv_detect_status
    actions["tv_detect_on"]        = tv_detect_on
    actions["tv_detect_off"]       = tv_detect_off
    print("  [tv-detect] camera TV-on detector actions registered "
          "(opt-in via TV_DETECT_ENABLED, off by default)")
