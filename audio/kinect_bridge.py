"""
kinect_bridge — lazy, graceful Xbox Kinect v2 client.

WHY THIS MODULE EXISTS
======================
The Kinect v2 (`pykinect2` + the Kinect Runtime) is a heavyweight, Windows-only,
single-consumer sensor. Touching it at import time would (a) drag the comtypes /
ctypes Kinect SDK bindings into every JARVIS boot even on machines with no
sensor, and (b) `pykinect2` 0.1.0 does not even *import* unmodified on
Python 3.14. So this bridge mirrors audio/itunes_bridge.py:

  • NOTHING Kinect-related is imported at module load. Every `import` lives
    inside a function, after the enabled-gate. Importing this bridge from
    anywhere — bobert_companion, the face-tracker skill, a voice action — costs
    only a few function definitions.

  • A single PyKinectRuntime is opened lazily on first use and cached behind a
    threading.Lock (the runtime is NOT safe to open twice — the second open
    fails to bind the sensor). All public accessors share that one runtime.

  • Every public accessor returns a graceful sentinel (None / [] / a "not
    available" dict) and NEVER raises to the caller. A missing sensor, an
    absent dependency, or a mid-stream COM hiccup degrades to "I can't see
    through the Kinect right now, sir" rather than crashing the voice loop.

PYKINECT2 ON PYTHON 3.14
========================
`pykinect2` 0.1.0 assumes an older Python/numpy: it calls `time.clock()`
(removed in 3.12), references `numpy.object` (removed in numpy 1.24+), and has
a couple of `assert sizeof(...)` / `_check_version(...)` lines that abort import
on a mismatched comtypes. Rather than edit the installed package on disk (which
a pip reinstall or a fresh machine would wipe), `import_pykinect2()` reads the
package source, regex-patches the offending lines IN MEMORY, and execs the
patched source into freshly-created modules registered in sys.modules. This
exact loader is proven to import and stream on this machine.

CONFIGURATION
=============
`set_enabled(bool)` mirrors core.config.KINECT_ENABLED into the bridge (the same
pattern as itunes_bridge.set_auto_launch). When disabled — the privacy-conscious
default — `get_runtime()` and every accessor short-circuit to the graceful
sentinel WITHOUT opening the sensor. bobert_companion calls set_enabled at
startup next to the other bridge config hooks.

DROP-IN CAPTURE SHIM
====================
`KinectCapture` exposes `.read() -> (ret, bgr_frame)` and `.release()` with the
same shape cv2.VideoCapture gives, so the monolith's _open_capture() can return
one in place of a cv2.VideoCapture and the rest of _face_tracking_thread keeps
working unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
import threading
import time
from typing import Any, Optional


# ─── the proven pykinect2 in-memory patch-loader ──────────────────────────
# Do NOT edit the installed package on disk — patch the source string and
# exec it into fresh modules. This is the EXACT loader validated live on this
# machine (see module docstring).

def _load_patched(name: str, subs):
    """Import `name` from its real source with the (pattern, replacement)
    regex substitutions in `subs` applied to the source first. Cached via
    sys.modules so a second call returns the already-execed module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ImportError(name)
    src = open(spec.origin, encoding="utf-8", errors="replace").read()
    for pat, rep in subs:
        src = re.sub(pat, rep, src, flags=re.M)
    mod = importlib.util.module_from_spec(spec)
    # Insert into sys.modules ONLY after a clean exec. The old code registered the
    # half-built module BEFORE exec, so if exec raised (a patch that left the source
    # un-runnable) the broken module stuck in sys.modules and EVERY later
    # import_pykinect2() returned it via the cache short-circuit above — poisoning
    # the whole bridge after one bad load. Register-after-exec; on failure drop any
    # partial entry and re-raise as ImportError so the next call re-attempts cleanly.
    try:
        exec(compile(src, spec.origin, "exec"), mod.__dict__)
    except Exception as e:
        sys.modules.pop(name, None)
        raise ImportError(f"{name}: patched exec failed: "
                          f"{type(e).__name__}: {e}") from e
    sys.modules[name] = mod
    return mod


def import_pykinect2():
    """Return (PyKinectV2, PyKinectRuntime) modules, patched to import on
    Python 3.14 + modern numpy. Raises ImportError if pykinect2 isn't
    installed. Idempotent (sys.modules caches the patched modules)."""
    # Run the package __init__ (harmless, no Kinect contact) so the two
    # submodules resolve. import_module instead of a bare `import pykinect2`
    # so pyflakes doesn't flag an "unused import".
    importlib.import_module("pykinect2")
    _load_patched("pykinect2.PyKinectV2", [
        (r"^(\s*)assert sizeof\(", r"\1pass  # pk2patch: assert sizeof("),
        (r"^(\s*)(.*_check_version\(.*)$", r"\1pass  # pk2patch: \2"),
    ])
    _load_patched("pykinect2.PyKinectRuntime", [
        (r"time\.clock\(", r"time.perf_counter("),
        (r"numpy\.object\b", r"object"),   # newer numpy removed np.object
    ])
    return sys.modules["pykinect2.PyKinectV2"], sys.modules["pykinect2.PyKinectRuntime"]


# ─── configuration hook ───────────────────────────────────────────────────

_ENABLED: bool = False


def set_enabled(enabled: bool) -> None:
    """Mirror core.config.KINECT_ENABLED into the bridge. When False (the
    privacy-conscious default) the sensor is never opened. Called by
    bobert_companion at startup, next to itunes_bridge.set_auto_launch."""
    global _ENABLED
    _ENABLED = bool(enabled)
    if not _ENABLED:
        # Opting out should also tear down a runtime opened by a prior
        # enabled session so the sensor LED goes dark immediately, and stop the
        # always-on body-frame pump (PART B).
        stop_body_pump()
        close()
    else:
        # Opting in starts the always-on body-frame pump (singleton-guarded) so
        # the body stream can't go quiet once warmed (PART B). Safe to call
        # repeatedly — it no-ops when a pump is already running.
        start_body_pump()


def get_enabled() -> bool:
    return _ENABLED


# ─── joint-name map (PyKinectV2.JointType_* indices → friendly names) ─────
# Hard-coded so get_bodies() can return readable joint keys without importing
# PyKinectV2 just to read its constants. Index order matches the SDK enum
# (verified against the installed PyKinectV2.py).
_JOINT_NAMES = (
    "spine_base", "spine_mid", "neck", "head",
    "shoulder_left", "elbow_left", "wrist_left", "hand_left",
    "shoulder_right", "elbow_right", "wrist_right", "hand_right",
    "hip_left", "knee_left", "ankle_left", "foot_left",
    "hip_right", "knee_right", "ankle_right", "foot_right",
    "spine_shoulder", "hand_tip_left", "thumb_left",
    "hand_tip_right", "thumb_right",
)
_JOINT_COUNT = len(_JOINT_NAMES)   # 25


# ─── hand-state map (PyKinectV2.HandState_* indices → friendly names) ──────
# The Kinect v2 reports a discrete OPEN/CLOSED/LASSO grip per hand alongside the
# skeleton (body.hand_right_state / hand_left_state — verified against the
# installed PyKinectRuntime, which sets these ints on each KinectBody). Mapped
# to lowercase strings here so callers (the air-mouse skill) read "open" /
# "closed" without importing PyKinectV2 just for its enum. Index order matches
# the SDK enum: 0 Unknown, 1 NotTracked, 2 Open, 3 Closed, 4 Lasso. We collapse
# NotTracked → "unknown" since both mean "no reliable grip this frame".
_HAND_STATE_NAMES = {
    0: "unknown",   # HandState_Unknown
    1: "unknown",   # HandState_NotTracked (no usable grip → treat as unknown)
    2: "open",      # HandState_Open
    3: "closed",    # HandState_Closed
    4: "lasso",     # HandState_Lasso (two-finger "pointer"; not used by v1)
}


def _hand_state_name(raw: Any) -> str:
    """Map a raw Kinect hand-state int to a friendly lowercase name. Anything
    unexpected (None, out-of-range, non-int) degrades to "unknown" — this is a
    pure helper that, like the rest of the bridge, never raises."""
    try:
        return _HAND_STATE_NAMES.get(int(raw), "unknown")
    except (TypeError, ValueError):
        return "unknown"


# ─── singleton runtime (cached behind a lock) ─────────────────────────────
# Module-list wrapping so the lock-guarded mutators can reassign without a
# `global`. _runtime[0] is the live PyKinectRuntime (or None). _negative_until
# briefly caches an "unavailable" verdict so available() doesn't re-probe the
# SDK on every call when there's no sensor.
_lock = threading.RLock()
_runtime: list[Any] = [None]
_open_error: list[Optional[str]] = [None]
_negative_until = [0.0]            # monotonic; available() negative-cache expiry
# Shortened from 30s: the sensor re-arrival latency (the owner plugging the Kinect
# back in / a prior holder releasing it) should be felt within a few seconds, not
# blocked for half a minute. clear_negative_cache() lets a caller bypass it
# outright when it knows the sensor just (re)appeared.
_NEGATIVE_CACHE_SEC = 5.0
# A just-hard-killed prior instance can still hold the sensor when we reopen;
# PyKinectRuntime() then returns a live object that never streams a frame.
# Verify real frames arrive and retry so a restart can't latch a dead runtime.
_OPEN_STREAM_RETRIES = 4
_OPEN_STREAM_RETRY_SEC = 1.5
# Longer negative cache specifically for a WEDGED sensor (opened but streamed
# no frames through the full retry gauntlet — e.g. held by KStudioHostService
# or a USB-stalled stack). The full gauntlet costs tens of seconds; live
# 2026-07-10 it ran on the VOICE loop for camera_status/air_mouse_on and
# tripped the main-loop watchdog THREE times (~60-70s stalls each). Once it
# has failed completely, fail FAST for this long before trying again.
# clear_negative_cache() still bypasses it (device-refresh / replug paths).
_WEDGED_CACHE_SEC = 90.0
# Guards the UNLOCKED open/verify/retry work so two callers don't both pound the
# (single-consumer) sensor with concurrent PyKinectRuntime() opens. The publish of
# the winning runtime still happens under _lock; this only serialises the slow
# verify so the loser waits for the winner instead of racing it (M2).
_open_attempt_lock = threading.Lock()


def clear_negative_cache() -> None:
    """Forget any cached 'sensor unavailable' verdict so the very next available()/
    get_runtime() re-probes the SDK immediately instead of waiting out the negative
    cache. Call this when the sensor is known to have just (re)appeared. NEVER
    raises."""
    _negative_until[0] = 0.0
    _open_error[0] = None


def _frame_source_flags(pk2):
    """Color | Body | Depth | Infrared — the full set the bridge streams."""
    return (pk2.FrameSourceTypes_Color | pk2.FrameSourceTypes_Body
            | pk2.FrameSourceTypes_Depth | pk2.FrameSourceTypes_Infrared)


def _runtime_streams(rt, timeout_sec: float = 2.5,
                     require_color: bool = True) -> bool:
    """True if a freshly-opened runtime delivers the required frame(s) within
    timeout_sec. A sensor still held by a releasing prior instance opens but never
    streams — catch that so we retry instead of caching a dead runtime.

    require_color (the preview-keep-alive fix): when True we require a COLOR frame
    specifically (the preview is color-backed), so a reopen that yields BODY-but-no-
    COLOR is REJECTED and retried rather than cached as "good" — which is exactly
    how the old reopen left color cold (it accepted body-only) and the skeleton
    stayed dark while gestures worked. We still also accept color when body lags, so
    a color-live runtime always passes. require_color=False keeps the old
    color-OR-body behaviour for callers that don't need the preview."""
    end = time.monotonic() + timeout_sec
    saw_body = False
    while time.monotonic() < end:
        try:
            if rt.has_new_color_frame():
                return True
            # Track body so a no-color (require_color False) caller still passes.
            if rt.has_new_body_frame():
                saw_body = True
                if not require_color:
                    return True
        except Exception:
            return False
        time.sleep(0.05)
    # Timed out without a color frame. Only honour a body-only stream when the
    # caller didn't require color (require_color True → reject so the open retries).
    return saw_body and not require_color


def _safe_close_runtime(rt) -> None:
    try:
        close = getattr(rt, "close", None)
        if callable(close):
            close()
    except Exception:  # pragma: no cover - defensive: older builds lack close()
        pass


def _publish_runtime(rt) -> Any:
    """Adopt `rt` as the shared runtime under _lock, or — if another caller already
    published one while we were verifying (the race a second accessor loses) — close
    OUR loser and return the established winner. Returns the live runtime that is now
    in _runtime[0]. Seeds both staleness clocks + starts the body pump on a genuine
    fresh open (M1). NEVER raises."""
    with _lock:
        if _runtime[0] is not None and _runtime[0] is not rt:
            winner = _runtime[0]
            adopt_loser = True
        else:
            _runtime[0] = rt
            _open_error[0] = None
            # Seed BOTH staleness clocks so the freshly-verified runtime gets a full
            # window to stream before any stale-reset could fire (the verify already
            # confirmed a color frame, so color is live right now).
            now0 = time.monotonic()
            _last_body_frame_at[0] = now0
            _last_color_frame_at[0] = now0
            # Seed the per-stream seen-cells from THIS runtime's current frame
            # times: a reopened instance must neither register a spurious "new
            # frame" off its just-verified first frames nor be masked by the dead
            # instance's older perf_counter stamps (perf_counter is process-wide,
            # so cross-instance comparisons are meaningless).
            for _attr, _cell in (("_last_color_frame_time", _color_time_seen),
                                 ("_last_body_frame_time", _body_time_seen)):
                try:
                    _cell[0] = float(getattr(rt, _attr, 0.0) or 0.0)
                except (TypeError, ValueError):
                    _cell[0] = 0.0
            winner = rt
            adopt_loser = False
    if adopt_loser:
        # A peer won the race; release our redundant handle so the sensor isn't
        # held twice. Outside the lock — close() can block on a half-dead handle.
        _safe_close_runtime(rt)
    else:
        # Fresh open → make sure the always-on body pump is running so the body pipe
        # is kept warm (it only flows while something reads it). start_body_pump is
        # singleton-guarded + no-ops when disabled, so this is safe to call here as
        # well as from set_enabled(True) (M1).
        _ensure_pump_alive()
    return winner


def _open_runtime_locked():
    """Open (or return the cached) PyKinectRuntime. Returns (runtime, None) or
    (None, reason).

    Despite the name (kept for call-site stability) this NO LONGER does its slow
    work under _lock (M2): the old version held _lock across up to 2.5 s of stream
    verification × 4 attempts × 1.5 s sleeps (~16 s worst case), blocking EVERY
    accessor and the voice loop on a single hiccup. Now the verify/retry/sleep runs
    UNLOCKED (serialised only against other openers by _open_attempt_lock) and we
    re-acquire _lock just to publish the winning runtime — so a concurrent
    get_presence()/get_bodies() that finds _runtime[0] already set never waits on
    the opener at all."""
    # Fast path: already open. A plain read of the single-cell list is atomic under
    # the GIL, so this common case needs no lock.
    rt0 = _runtime[0]
    if rt0 is not None:
        return rt0, None
    if not _ENABLED:
        return None, ("Kinect is disabled — set KINECT_ENABLED = True to "
                      "enable (it's off by default for privacy).")
    # Serialise the slow open so two callers don't both hammer the single-consumer
    # sensor with concurrent opens. Whoever loses the lock re-checks the cache on
    # entry and rides the winner's published runtime.
    #
    # BOUNDED ACQUIRE (2026-07-14 audit). This was a plain `with
    # _open_attempt_lock:`. The locked body runs a retry gauntlet —
    # _OPEN_STREAM_RETRIES (4) attempts, each a 2.5 s stream-verify plus a 1.5 s
    # sleep — so a WEDGED sensor holds the lock for up to ~16 s. The always-on
    # 30 Hz body pump calls get_runtime() and enters that gauntlet; when the
    # owner then says "camera status" the VOICE THREAD's get_runtime() blocked on
    # this acquire for the rest of the gauntlet, because the negative cache that
    # would fast-fail it isn't published until the FIRST failed open completes.
    # A second caller now fails fast instead: the pump will publish the runtime
    # (or the negative verdict) shortly, and the next call rides it.
    got = _open_attempt_lock.acquire(timeout=0.5)
    if not got:
        rt0 = _runtime[0]
        if rt0 is not None:
            return rt0, None
        return None, (_open_error[0] or "Kinect open already in progress")
    try:
        rt0 = _runtime[0]
        if rt0 is not None:
            return rt0, None
        try:
            pk2, rt_mod = import_pykinect2()
        except ImportError:
            return None, "pykinect2 not installed — pip install pykinect2"
        except Exception as e:   # pragma: no cover - patch-loader compile/exec failure
            return None, f"pykinect2 failed to load: {type(e).__name__}: {e}"
        last = None
        for _attempt in range(_OPEN_STREAM_RETRIES):
            try:
                rt = rt_mod.PyKinectRuntime(_frame_source_flags(pk2))
            except Exception as e:
                return None, f"could not open Kinect sensor: {type(e).__name__}: {e}"
            if _runtime_streams(rt):
                winner = _publish_runtime(rt)
                if _attempt:
                    print(f"  [kinect] sensor live after {_attempt + 1} open attempts")
                return winner, None
            last = "opened but no frames streaming"
            _safe_close_runtime(rt)
            time.sleep(_OPEN_STREAM_RETRY_SEC)
        return None, (f"Kinect opened but streamed no frames after "
                      f"{_OPEN_STREAM_RETRIES} attempts ({last}); sensor may be "
                      f"held by another process")
    finally:
        _open_attempt_lock.release()


def get_runtime() -> tuple[Any, Optional[str]]:
    """Return (PyKinectRuntime, None) with Color|Body|Depth|Infrared open, or
    (None, reason) if disabled / unavailable. Never raises.

    No longer wraps the open in _lock: _open_runtime_locked() now does its own fine-
    grained locking (a brief publish under _lock, the slow verify under a separate
    open-attempt lock) so this accessor no longer blocks every other caller for the
    whole open (M2).

    NEGATIVE-CACHED (2026-07-10): available() already failed fast after a bad
    open, but get_runtime() — what the ACTION handlers call — re-ran the full
    open/verify/retry gauntlet on every call. With a wedged sensor (held by
    KStudioHostService / USB-stalled: opens but never streams) that put tens of
    seconds on the VOICE loop per kinect-touching action and tripped the
    main-loop watchdog repeatedly. Now a completed-but-failed open is remembered
    for _WEDGED_CACHE_SEC and callers get the honest failure instantly."""
    rt0 = _runtime[0]
    if rt0 is not None:
        return rt0, None
    now = time.monotonic()
    if now < _negative_until[0] and _open_error[0]:
        return None, _open_error[0]
    rt, err = _open_runtime_locked()
    if rt is None and err:
        _publish_open_failure(err)
    return rt, err


def _publish_open_failure(err: str) -> None:
    """Record a failed open into the SHARED negative-cache cells, picking the
    right cooldown. Both get_runtime() and available() go through here so the
    cooldown policy can't diverge between them.

    This existed inline in get_runtime() only; available() had its own copy that
    ALWAYS wrote the short _NEGATIVE_CACHE_SEC — so on a WEDGED sensor (opens but
    streams no frames) an available() call (fired ~30 Hz by the air-mouse /
    two-hand pollers, and on the voice thread by kinect_status / who_is_here /
    air_mouse_on) would overwrite get_runtime()'s intended 90 s wedged cooldown
    with 5 s, re-arming the ~16 s open gauntlet every few seconds and letting it
    land back on the voice loop. Disabled-by-config is cheap to re-check, so it
    earns no cooldown; only a real failed open (import/open error, or the
    expensive wedged no-frames verify) does. 2026-07-14 bug-hunt."""
    _open_error[0] = err
    if "disabled" in err:
        return
    cool = _WEDGED_CACHE_SEC if "no frames" in err else _NEGATIVE_CACHE_SEC
    _negative_until[0] = time.monotonic() + cool


def available() -> tuple[bool, str]:
    """(True, "") if pykinect2 is importable AND a sensor opens; else
    (False, reason). The negative verdict is cached briefly so callers (the
    presence poller, kinect_status) don't re-probe the SDK every call when no
    sensor is attached. A positive result is NOT cached here — the live
    runtime in _runtime[0] is the cache."""
    if _runtime[0] is not None:
        # Sensor is open. Make sure the body pump that keeps the body pipe warm is
        # actually alive — if its thread died, restart it so we don't sit on an open
        # runtime whose body stream silently quiesced with nothing reading it (H2).
        _ensure_pump_alive()
        return True, ""
    now = time.monotonic()
    if now < _negative_until[0] and _open_error[0]:
        return False, _open_error[0]
    rt, err = _open_runtime_locked()
    if rt is not None:
        return True, ""
    # Shared cooldown policy — a wedged (no-frames) open earns the long
    # _WEDGED_CACHE_SEC here too, so an available() call can't stomp the 90 s
    # cooldown get_runtime() set down to 5 s. 2026-07-14 bug-hunt.
    _publish_open_failure(err or "Kinect unavailable")
    return False, _open_error[0]


# ─── frame accessors ──────────────────────────────────────────────────────
# Each grabs the latest frame if the sensor has one new, reshapes it, and
# returns a numpy array (or None). numpy/cv2 are imported lazily inside the
# functions so module import stays dependency-free on a sensorless / CI host.

# BRIDGE-DERIVED FRESHNESS (2026-07-21 audit: "staleness/self-heal is a no-op").
# The installed pykinect2 build assigns `_last_color_frame_access` /
# `_last_body_frame_access` as bare LOCALS inside its get_last_* methods (no
# `self.`), so has_new_color_frame()/has_new_body_frame() — which compare the
# advancing `self._last_*_frame_time` against the never-updated __init__ access
# stamp — are permanently True once ONE frame of that stream has ever arrived.
# Everything the bridge built on those flags (the `had_new` color stamp, the
# body `pending` gate, both staleness clocks, and thus reset_if_body_stale) was
# therefore a no-op on a wedged/unplugged sensor: frozen frames served forever,
# presence asserting a departed person, no self-heal reopen. The
# `_last_*_frame_time` attrs ARE trustworthy — handle_*_arrived() advances them
# only on real frame arrival — so the bridge now tracks its own last-consumed
# frame time per stream and derives "a genuinely new frame is pending" from the
# time ADVANCING past that stamp. On runtimes without the timestamp attrs (test
# fakes / foreign builds) the helper degrades to the has_new_* flag, keeping
# clear-on-read builds working unchanged.
_color_time_seen = [0.0]           # last _last_color_frame_time this bridge served
_body_time_seen = [0.0]            # last _last_body_frame_time this bridge consumed


def _frame_time_advanced(rt, time_attr: str, flag_attr: str,
                         cell: list) -> tuple[bool, Optional[float]]:
    """(advanced, t) for one stream: `advanced` is True iff a GENUINELY new
    frame is pending — the runtime's `time_attr` (e.g. _last_color_frame_time)
    has advanced past `cell[0]`, the last value this bridge consumed. `t` is the
    runtime's current frame time so the CALLER can stamp `cell[0] = t` when it
    actually serves/consumes the frame (None when the attr is absent/unusable,
    in which case the verdict came from the `flag_attr` readiness flag fallback
    and there is nothing to stamp). NEVER raises."""
    try:
        t = getattr(rt, time_attr, None)
        if t is not None:
            try:
                t = float(t)
            except (TypeError, ValueError):
                t = None
        if t is None:
            # No usable timestamp on this build → trust the readiness flag
            # (correct on clear-on-read builds and the test fakes).
            flag = getattr(rt, flag_attr, None)
            return (bool(flag()) if callable(flag) else False), None
        return (t > cell[0]), t
    except Exception:   # pragma: no cover - defensive: odd runtime attr
        return False, None


def get_color_bgr(require_new: bool = True):
    """Latest color frame as a (1080, 1920, 3) BGR uint8 ndarray, or None.

    The Kinect delivers a flat uint8 of length 8294400 = 1920*1080*4 in BGRA
    order; we reshape to (1080,1920,4) and drop the alpha → BGR (what cv2
    expects).

    require_new=True (default) returns None when no frame has arrived since the
    last call — right for one-shot callers (ask_vision, get_color_png). The
    KinectCapture shim passes require_new=False so a poll faster than the
    sensor's ~30 fps still yields the most recent frame instead of a spurious
    read-failure (the monolith's face-track loop treats None as a dropped
    frame and escalates to a webcam-reopen)."""
    rt, _ = get_runtime()
    if rt is None:
        return None
    try:
        import numpy as np
        # Capture whether a genuinely NEW color frame is pending BEFORE we read the
        # buffer (H1). get_last_color_frame() re-serves the SAME buffer for a
        # require_new=False peek even when nothing new arrived, so stamping the color
        # clock on every byte-returning call (as the old code did) kept
        # _last_color_frame_at fresh forever — including on the ~30 Hz pump's
        # require_new=False prime — so "color is stale" could NEVER become true on a
        # fully-dead sensor and the both-plane stale-reset never fired. Stamp ONLY
        # when had_new; still SERVE the re-served buffer for require_new=False peeks.
        # `had_new` is BRIDGE-DERIVED from _last_color_frame_time advancing (see
        # _frame_time_advanced): the installed build's has_new_color_frame() is
        # permanently True after the first frame ever, which silently defeated
        # this whole gate (2026-07-21 audit).
        had_new, ct = _frame_time_advanced(
            rt, "_last_color_frame_time", "has_new_color_frame", _color_time_seen)
        if require_new and not had_new:
            return None
        flat = rt.get_last_color_frame()
        if flat is None:
            return None
        arr = np.asarray(flat, dtype=np.uint8)
        if arr.size != 1920 * 1080 * 4:
            return None
        bgra = arr.reshape((1080, 1920, 4))
        # Stamp the COLOR staleness clock ONLY on a real new frame (the preview-keep-
        # alive fix): a re-served stale buffer must NOT count as the color stream
        # being live, or the both-plane stale-reset can never fire (H1). Also stamp
        # the seen-cell in BOTH require_new modes (the ~30 Hz pump prime is the
        # usual warmer) so a frozen frame time reads as consumed exactly once.
        if had_new:
            if ct is not None:
                _color_time_seen[0] = ct
            note_color_frame_seen()
        return bgra[:, :, :3]   # BGRA → BGR (drop alpha)
    except Exception:   # pragma: no cover - defensive: mid-stream frame glitch
        return None


def get_color_png():
    """Latest color frame encoded as PNG bytes (for ask_vision), or None."""
    bgr = get_color_bgr()
    if bgr is None:
        return None
    try:
        import cv2
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return None
        return bytes(buf.tobytes())
    except Exception:   # pragma: no cover - defensive: cv2 encode failure
        return None


def get_infrared_gray():
    """Latest infrared frame as an 8-bit (424, 512) grayscale ndarray for
    night-vision, or None. IR arrives as 512*424 uint16; we normalise to
    uint8 so it's directly viewable / encodable.

    NB: the installed pykinect2 0.1.0 build does NOT actually wire up the
    infrared stream — its __init__ never subscribes an IR reader and
    handle_infrared_arrived() is a stub, so there is no get_last_infrared_frame
    getter and no IR buffer to read. This therefore returns None on this build
    (verified live). The accessor is kept (and reads via getattr) so it starts
    working automatically if a fuller pykinect2 build that exposes IR is later
    installed — color, depth, and body all work today."""
    rt, _ = get_runtime()
    if rt is None:
        return None
    try:
        import numpy as np
        # Probe BOTH the readiness check AND the getter defensively (M4): this build
        # lacks get_last_infrared_frame entirely, and a build that wires one but not
        # the other would have made the old code (which probed only the getter but
        # called has_new_infrared_frame directly) AttributeError into the broad
        # except — indistinguishable from "no frame". Bail cleanly if EITHER is
        # missing so a partial IR build degrades to None rather than a swallowed
        # raise.
        getter = getattr(rt, "get_last_infrared_frame", None)
        has_new = getattr(rt, "has_new_infrared_frame", None)
        if not callable(getter) or not callable(has_new):
            return None
        if not has_new():
            return None
        flat = getter()
        if flat is None:
            return None
        arr = np.asarray(flat, dtype=np.uint16)
        if arr.size != 512 * 424:
            return None
        frame = arr.reshape((424, 512))
        # Normalise the 16-bit IR to 8-bit. A fixed >>8 crushes the contrast
        # (IR rarely uses the top byte), so scale by the actual max.
        peak = int(frame.max())
        if peak <= 0:
            return frame.astype(np.uint8)
        scaled = (frame.astype(np.float32) * (255.0 / peak))
        return scaled.clip(0, 255).astype(np.uint8)
    except Exception:   # pragma: no cover - defensive: mid-stream frame glitch
        return None


def get_depth():
    """Latest depth frame as a (424, 512) uint16 ndarray (millimetre-ish
    depth), or None."""
    rt, _ = get_runtime()
    if rt is None:
        return None
    try:
        import numpy as np
        if not rt.has_new_depth_frame():
            return None
        flat = rt.get_last_depth_frame()
        if flat is None:
            return None
        arr = np.asarray(flat, dtype=np.uint16)
        if arr.size != 512 * 424:
            return None
        return arr.reshape((424, 512))
    except Exception:   # pragma: no cover - defensive: mid-stream frame glitch
        return None


# ─── body / skeleton tracking ─────────────────────────────────────────────
# SHARED BODY-FRAME CACHE (see get_bodies()): the body PUMP is the single reader
# of the sensor's single-consumer body frame; it parses bodies ONCE per frame
# into _body_cache with a monotonic stamp in _body_cache_at, and every consumer
# reads the cache (no per-frame competition). _body_cache is None until the first
# successful read; [] is a valid cached "no bodies tracked" value (distinct from
# None = never populated). BODY_CACHE_FRESH_SEC bounds how old a cache entry may
# be and still be served before a consumer falls back to a direct read.
BODY_CACHE_FRESH_SEC = 0.30        # serve cache stamped within this window
# The pump-alive raw-cache fallback (in _read_and_cache_bodies) exists to bridge
# the ~33 ms one-tick race with the pump, NOT to keep serving a body the pump
# dropped seconds ago when it stalls (GC / CUDA contention — all documented on
# this box). Bound that peek by age so a stalled-but-alive pump degrades to "no
# one present" instead of a departed-person phantom. Modest so it still absorbs
# the intended one-tick race; deliberately NOT BODY_STALE_RESET_SEC (4 s, far too
# loose for a presence/grip signal). (2026-07-15 ghost audit — CONFIRMED vector)
_RAW_CACHE_FALLBACK_MAX_SEC = 0.5
_body_cache: list[Any] = [None]    # last parsed bodies (None=cold, []=none tracked)
_body_cache_at = [0.0]             # monotonic stamp of the last cache populate
_body_cache_lock = threading.Lock()


def _store_bodies_cache(bodies: list, now: Optional[float] = None) -> None:
    """Publish a freshly-parsed body list to the shared cache with a monotonic
    stamp. Holds a tiny lock so a consumer reading the two cells never sees a
    torn (bodies, stamp) pair. NEVER raises."""
    ts = time.monotonic() if now is None else now
    with _body_cache_lock:
        _body_cache[0] = bodies
        _body_cache_at[0] = ts


def _get_cached_bodies(now: Optional[float] = None) -> Optional[list]:
    """Return the cached bodies if they were stamped within BODY_CACHE_FRESH_SEC,
    else None (cold or stale → the caller does a one-shot read). Returns a SHALLOW
    COPY of the list so a consumer can't mutate the shared cache; the per-body
    dicts are read-only by contract. NEVER raises."""
    n = time.monotonic() if now is None else now
    with _body_cache_lock:
        bodies = _body_cache[0]
        ts = _body_cache_at[0]
    if bodies is None:
        return None
    if (n - ts) > BODY_CACHE_FRESH_SEC:
        return None
    return list(bodies)


def _read_and_cache_bodies(consume: bool = True) -> list[dict]:
    """The body-frame reader. NEVER raises.

    consume=True (the PUMP only): when a new body frame is pending, read it ONCE
    (which CLEARS has_new_body_frame), parse it, publish to the shared cache, and
    stamp the staleness clock; return the freshly-parsed bodies. This is the SOLE
    path that actually consumes the sensor's single-consumer body frame.

    consume=False (get_bodies()'s cold/stale FALLBACK): a non-pump consumer must NOT
    steal the body frame from the pump + the other consumers (H3). The body frame is
    single-consumer — get_last_body_frame() clears the new-frame flag — so a
    consumer that consumed it here would blank the pump and every other poller that
    tick. So when consume=False: if the PUMP is alive, NEVER clear the flag — serve
    the cache (even marginally stale: the pump will refresh it within ~33 ms) and
    leave the pending frame for the pump. Only when NO pump is alive (a lone consumer
    with nothing else reading) does consume=False fall through to a real read, so a
    pump-less caller still isn't left blind.

    When no NEW frame is pending either way, return the last cached bodies if still
    fresh (so a direct caller racing the pump still gets data), else []."""
    rt, _ = get_runtime()
    if rt is None:
        return []
    try:
        # BRIDGE-DERIVED pending (2026-07-21 audit): the installed build's
        # has_new_body_frame() is permanently True after the first frame ever
        # (bare-local access-stamp bug), which made this gate re-parse the frozen
        # _body_frame_bodies and re-stamp the staleness clock forever on a wedged
        # sensor. Derive "a new frame is pending" from _last_body_frame_time
        # advancing past what we last consumed; degrade to the flag on builds
        # without the timestamp attr (see _frame_time_advanced).
        pending, bt = _frame_time_advanced(
            rt, "_last_body_frame_time", "has_new_body_frame", _body_time_seen)
        if not pending:
            # No new frame to consume this tick — serve a still-fresh cache so two
            # readers in the same frame don't see []; else nothing to report.
            fresh = _get_cached_bodies()
            return fresh if fresh is not None else []
        if not consume and _pump_is_alive():
            # A new frame IS pending but we're a non-consuming fallback and the pump
            # is alive: do NOT steal the frame. Serve the cache if we have anything
            # at all (even past the freshness window — the pump refreshes it next
            # tick); only fall to [] when the cache is genuinely cold. We deliberately
            # peek the raw cache cell here (not _get_cached_bodies, which drops a
            # stale entry) so a consumer racing one tick ahead of the pump still gets
            # the last bodies instead of a spurious [].
            with _body_cache_lock:
                cached = _body_cache[0]
                ts = _body_cache_at[0]
            # AGE-BOUND the raw peek: fine to serve the last bodies for the one-tick
            # race, but a stalled pump must not keep asserting a departed person for
            # seconds. Past the cap, report "no bodies" rather than a phantom.
            if cached is not None and (time.monotonic() - ts) <= _RAW_CACHE_FALLBACK_MAX_SEC:
                return list(cached)
            return []
        # We ARE going to consume the frame (the pump, or a pump-less lone consumer).
        # Stamp the seen-cell ONLY on this consume path (the H3 non-consuming
        # fallback above leaves the frame — and the cell — pending for the pump)
        # and the staleness clock (PART B) so an actively-read, healthy stream
        # never trips the stale-reset, then take the single frame (clears the flag).
        if bt is not None:
            _body_time_seen[0] = bt
        note_body_frame_seen()
        frame = rt.get_last_body_frame()
    except Exception:   # pragma: no cover - defensive: mid-stream readiness/getter glitch
        return []
    bodies = _parse_body_frame(frame)
    _store_bodies_cache(bodies)
    return bodies


def _joint_distance(joints: dict) -> Optional[float]:
    """Best available z-distance for a body, in metres, or None. Prefers a
    FULLY-TRACKED (state>=2, finite, non-zero-fill) core joint and only accepts a
    plausible human range (0.5-4.5 m).

    GHOST FIX (2026-07-15): the old version returned any joint's z as long as
    `z > 0`, so an Inferred head at z=0.25 m — or an edge-noise blob at z=7 m —
    became a real distance and sorted the phantom as the NEAREST body ahead of the
    real user. Gating on _joint_reliable + the Kinect v2 body range denies that.
    Falls back down the preference list, so a real person with an inferred head
    still ranges off a tracked spine."""
    for name in ("head", "spine_shoulder", "spine_mid", "spine_base", "neck"):
        j = joints.get(name)
        if _joint_reliable(j):
            z = float(j[2])
            if 0.5 <= z <= 4.5:
                return z
    return None


# ─── arm-extension / forward-reach geometry (the air-mouse engage signal) ───
# The NEW air-mouse engages on a deliberate REACH — an arm extended OUT toward
# the sensor — not on a hand merely raised. Two independent cues describe that
# reach, both read off the SAME tracked skeleton the rest of the stack uses:
#
#   • FORWARD-DEPTH: the hand pushed forward of the torso in DEPTH. Kinect z
#     increases AWAY from the sensor, so a hand reaching toward the sensor has a
#     SMALLER z than the body. forward_reach = body_z - hand_z (positive when the
#     hand is in front of the body); the body reference is the spine_mid / spine_
#     shoulder, falling back to the same-side shoulder.
#   • ARM-STRAIGHTNESS: the shoulder→hand straight-line 3D distance as a fraction
#     of the arm's full length (shoulder→elbow + elbow→hand). A relaxed/bent arm
#     folds the forearm back so the straight-line distance is well short of the
#     summed bone length; a straightened reach makes them nearly equal (ratio →
#     1). Using the RATIO (not an absolute metre distance) makes it body-size
#     independent — it works for a long or short arm without per-user calibration.
#
# Both are returned per hand so the air-mouse controller can apply its own
# engage/disengage hysteresis on whichever cue(s) it wants and pick the more-
# extended arm to drive the cursor. This helper is PURE (joint dict in, numbers
# out) and NEVER raises — a missing/untracked joint degrades that field to None.

def _dist3(a, b) -> Optional[float]:
    """Euclidean 3D distance between two (x, y, z, ...) joint tuples, or None if
    either is missing / too short. Pure; never raises."""
    try:
        if not a or not b or len(a) < 3 or len(b) < 3:
            return None
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    except (TypeError, ValueError):
        return None


def _body_scale_m(joints: dict) -> Optional[float]:
    """A BODY-SIZE reference distance in metres for normalising the forward reach
    into a dimensionless, POSITION-INDEPENDENT ratio (the fix for "engage depended
    on how far the owner sat / where the chair was").

    Prefer the SHOULDER WIDTH (3D distance shoulder_left↔shoulder_right) — the most
    stable horizontal body span and the one least affected by raising an arm — and
    fall back to the TORSO HEIGHT (spine_base→spine_shoulder) when a shoulder is
    untracked. Both scale with the SAME body, so forward_reach_m / body_scale is a
    fraction of the owner's own frame: a relaxed hand sits near 0, a full reach near
    ~1+, INDEPENDENT of the absolute metres (which shrink as the owner sits back).
    Returns metres, or None when neither span is usable. Pure; never raises."""
    try:
        sl = joints.get("shoulder_left")
        sr = joints.get("shoulder_right")
        width = _dist3(sl, sr)
        if width is not None and width > 0.10:   # a plausible shoulder span (m)
            return width
        base = joints.get("spine_base")
        top = joints.get("spine_shoulder") or joints.get("spine_mid")
        torso = _dist3(base, top)
        if torso is not None and torso > 0.10:
            return torso
    except (TypeError, ValueError, KeyError):
        return None
    return None


def arm_extension(joints: dict, side: str) -> dict:
    """Describe how EXTENDED one arm (`side` ∈ {"left","right"}) is, for the
    air-mouse reach-to-engage gate. Reads the shoulder / elbow / hand joints of
    that side plus a torso depth reference off `joints` (the get_bodies() shape).
    NEVER raises — any missing joint leaves its field None. Shape:

        {"side": str,
         "hand": (x, y, z, state) | None,     # the controlling hand joint
         "forward_reach_m": float | None,     # body_z - hand_z (>0 = reaching)
         "reach_ratio": float | None,         # forward_reach_m / body_scale (0..~1+)
         "body_scale_m": float | None,        # shoulder width (or torso height) (m)
         "straightness": float | None,        # 0..~1; chord / summed-bone length
         "shoulder_hand_m": float | None,     # straight-line shoulder→hand (m)
         "arm_len_m": float | None,           # shoulder→elbow + elbow→hand (m)
         "shoulder_ref_y": float | None,      # shoulder-line Y reference (camera-up)
         "lift_m": float | None}              # hand_y - shoulder_ref_y (>0 = raised)

    forward_reach_m is POSITIVE when the hand is pushed toward the sensor (in
    front of the torso). reach_ratio NORMALISES that absolute reach by a body-size
    span (shoulder width, fallback torso height) so it is POSITION-INDEPENDENT —
    invariant to how far the owner sits / where the chair is.

    lift_m is the HEIGHT of the hand above the shoulder line (hand_y minus a
    shoulder reference Y; Kinect camera-space y increases UPWARD). It is POSITIVE
    when the hand is raised AT/ABOVE the shoulder, NEGATIVE (≈ -0.3..-0.5 m) when a
    hand rests at desk/waist level. This is the air-mouse's PRIMARY engage gate
    (RAISE-HIGH to engage): body-relative, so it is invariant to rotation, chair
    position, and distance — a hand resting on the desk sits far below the shoulder
    so it never engages, while a hand raised to point at the screen sits at/above
    shoulder level so it does. The shoulder reference is spine_shoulder (the centre
    of the shoulder line, most stable) with the same-side shoulder as the fallback.

    straightness ≈ 1 when the arm is straightened out, and drops toward ~0.5-0.7
    when the elbow is bent and the hand pulled back; forward_reach_m / reach_ratio
    are retained as weak SECONDARY cues only (the height gate is primary).

    TRACKING-STATE FLOOR: lift_m (the PRIMARY gate) is computed ONLY when BOTH the
    controlling hand and the shoulder reference are FULLY TRACKED (TrackingState
    >= 2) with real finite non-zero-fill coords — otherwise it stays None and the
    gate fails safe (is_extended treats None as NOT extended). The demoted
    forward/straightness cues are NOT floored this way (they can't engage on their
    own), so they may still populate off inferred joints for the debug log."""
    out = {"side": side, "hand": None, "forward_reach_m": None,
           "reach_ratio": None, "body_scale_m": None,
           "straightness": None, "shoulder_hand_m": None, "arm_len_m": None,
           "shoulder_ref_y": None, "lift_m": None}
    try:
        shoulder = joints.get(f"shoulder_{side}")
        elbow = joints.get(f"elbow_{side}")
        hand = joints.get(f"hand_{side}") or joints.get(f"wrist_{side}")
        out["hand"] = hand
        # HEIGHT / LIFT (the PRIMARY gate): hand Y vs a shoulder-line Y reference.
        # Kinect camera-space y increases UPWARD, so lift_m > 0 means the hand is
        # at/above the shoulder (raised to point), and a hand resting on the desk
        # sits well below the shoulder (lift_m strongly negative).
        # TRACKING-STATE FLOOR (item 9): the lift gate is the air-mouse's PRIMARY
        # engage signal, so it must rest on RELIABLY-TRACKED joints only. Require
        # BOTH the controlling hand AND the shoulder reference to be fully TRACKED
        # (state >= 2) with real, non-zero-fill, finite coords before computing
        # lift_m; otherwise leave it None. An Inferred (1) or NotTracked (0,
        # zero-filled) joint is the SDK's guess and would fabricate a bogus "raised"
        # lift from noise — engaging the cursor off a hand the sensor can't actually
        # see. is_extended() already fails safe on lift_m is None (→ NOT extended),
        # so leaving it None here simply declines to engage rather than guessing.
        # Prefer spine_shoulder (the shoulder-line centre, steadiest + unaffected by
        # raising either arm); fall back to the same-side shoulder only if IT is
        # reliably tracked. Kinect camera-space y increases UPWARD.
        shoulder_ref = joints.get("spine_shoulder")
        if not _joint_reliable(shoulder_ref):
            shoulder_ref = shoulder if _joint_reliable(shoulder) else None
        if shoulder_ref is not None and _joint_reliable(hand):
            out["shoulder_ref_y"] = float(shoulder_ref[1])
            out["lift_m"] = float(hand[1]) - float(shoulder_ref[1])
        # FORWARD-DEPTH (weak secondary cue): hand z vs a torso depth reference
        # (spine first, then the same-side shoulder). Positive = hand in front.
        body_ref = None
        for name in ("spine_mid", "spine_shoulder", "spine_base"):
            j = joints.get(name)
            if j and len(j) >= 3 and float(j[2]) > 0:
                body_ref = j
                break
        if body_ref is None and shoulder and len(shoulder) >= 3:
            body_ref = shoulder
        if (body_ref is not None and hand and len(hand) >= 3
                and float(hand[2]) > 0 and float(body_ref[2]) > 0):
            out["forward_reach_m"] = float(body_ref[2]) - float(hand[2])
        # BODY-RELATIVE REACH RATIO (POSITION-INDEPENDENT): normalise the forward
        # reach by a body-size span so the engage/disengage gate is invariant to
        # the owner's distance from the sensor. shoulder width preferred, torso
        # height fallback (see _body_scale_m).
        scale = _body_scale_m(joints)
        out["body_scale_m"] = scale
        if (out["forward_reach_m"] is not None and scale is not None
                and scale > 1e-3):
            out["reach_ratio"] = out["forward_reach_m"] / scale
        # ARM-STRAIGHTNESS: shoulder→hand chord / (shoulder→elbow + elbow→hand).
        chord = _dist3(shoulder, hand)
        upper = _dist3(shoulder, elbow)
        fore = _dist3(elbow, hand)
        out["shoulder_hand_m"] = chord
        if upper is not None and fore is not None:
            arm_len = upper + fore
            out["arm_len_m"] = arm_len
            if chord is not None and arm_len > 1e-3:
                # Cap at 1.0: a near-straight arm can read marginally over 1 from
                # joint noise; clamp so the ratio is a clean 0..1 straightness.
                out["straightness"] = min(1.0, chord / arm_len)
        return out
    except (TypeError, ValueError, KeyError):
        return out


def _body_is_facing(joints: dict) -> Optional[bool]:
    """Rough 'is this body facing the sensor' heuristic, or None if we can't
    tell. We don't have HD-face orientation in scope, so approximate: the head
    is present AND sits above the spine (upright torso) AND both shoulders are
    roughly equidistant in z (chest toward the camera rather than side-on).

    FAIL-TO-NONE CONTRACT (2026-07-21 audit — the last un-migrated copy of the
    2026-07-15 ghost-audit joint-reliability gate): every joint read is gated on
    _joint_reliable, mirroring _joint_distance / _body_facing_yaw. A NotTracked
    zero-fill (0,0,0) head or shoulder used to fabricate a confident WRONG False
    (`0.0 > spine.y` reads as slumped; `|0.0 - z|` reads as side-on), which
    demoted the real owner to the worst facing_rank in the gesture owner-pick
    and misreported get_presence()['facing']. When the head/spine references
    aren't reliably tracked we return None (unknown) — never a guess; the
    downstream any()-aggregation and facing_rank sort already handle None. The
    shoulder z-gap term is used only when BOTH shoulders are reliable,
    otherwise the verdict falls back to upright-only."""
    head = joints.get("head")
    spine = joints.get("spine_shoulder")
    if not _joint_reliable(spine):
        # Reliability-aware fallback: a present-but-unreliable spine_shoulder
        # must not block fall-through to a reliable spine_mid.
        spine = joints.get("spine_mid")
    if not _joint_reliable(head) or not _joint_reliable(spine):
        return None
    # Kinect camera-space y increases UPWARD, so an upright person has
    # head.y > spine.y.
    upright = head[1] > spine[1]
    sl = joints.get("shoulder_left")
    sr = joints.get("shoulder_right")
    if _joint_reliable(sl) and _joint_reliable(sr):
        # Side-on bodies show a big z-gap between the two shoulders; facing
        # bodies show both shoulders at a similar depth.
        shoulder_facing = abs(float(sl[2]) - float(sr[2])) < 0.30
        return bool(upright and shoulder_facing)
    return bool(upright)


def _tracked(j) -> bool:
    """True when a joint tuple (x, y, z, tracking_state) is at least INFERRED.
    Kinect TrackingState: 0 = NotTracked, 1 = Inferred, 2 = Tracked. We accept
    >= 1 here (a position the SDK is willing to report) and let callers that
    need a firmer fix demand state >= 2 themselves."""
    return j is not None and len(j) >= 4 and int(j[3]) >= 1


# A joint is RELIABLE for the gating geometry (the air-mouse lift gate) only when
# its TrackingState is fully TRACKED (>= 2). Inferred (1) joints are the SDK's
# best GUESS for an occluded joint and read as noisy/zero-filled positions — using
# them for the engage gate let an inferred or NotTracked-zero-filled hand fabricate
# a "raised" lift and engage the cursor. Mirrors kinect_pointing._joint_ok /
# MIN_TRACKING_STATE = 2. Also rejects the SDK's NotTracked zero-fill (x==y==z==0.0)
# and any non-finite coordinate, both of which read as a spurious position.
_MIN_RELIABLE_TRACKING_STATE = 2


def _joint_reliable(j) -> bool:
    """True when a (x, y, z, tracking_state) joint is FULLY TRACKED (state >= 2)
    AND carries a real, finite, non-zero-fill position. False for a missing /
    too-short tuple, an Inferred/NotTracked joint, the NotTracked zero-fill
    (x==y==z==0.0), or any NaN/Inf coordinate. Pure; never raises."""
    try:
        if not j or len(j) < 4:
            return False
        if int(j[3]) < _MIN_RELIABLE_TRACKING_STATE:
            return False
        x, y, z = float(j[0]), float(j[1]), float(j[2])
        # NotTracked frames zero-fill the position; treat an exact (0,0,0) as
        # untracked rather than a hand sitting precisely on the sensor origin.
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False
        # Reject non-finite coords (NaN/Inf) that would poison the lift arithmetic.
        if not (x == x and y == y and z == z):   # NaN != itself
            return False
        if x in (float("inf"), float("-inf")) or y in (float("inf"), float("-inf")) \
                or z in (float("inf"), float("-inf")):
            return False
        return True
    except (TypeError, ValueError):
        return False


# ─── ghost-skeleton rejection (real body vs. inferred/zero-fill phantom) ────
# The Kinect v2 runtime marks a body slot is_tracked=True for reflections,
# furniture, a coat rack, edge-of-frame noise, or a person who JUST left frame,
# filling that skeleton with Inferred(1)/NotTracked(0, zero-filled) joints. That
# "ghost skeleton" (the owner's report: "sees ghost skeletons when arms aren't
# present") then inflates get_presence count, can be picked as the nearest body,
# and draws a phantom stick-figure on the HUD. A REAL person facing the sensor
# shows ~15-25 FULLY-TRACKED (state>=2) joints; a phantom shows 0-3. Requiring a
# modest floor of reliable joints INCLUDING a reliable core (spine/neck/head)
# drops the phantom at the SINGLE parse point every consumer reads, while a
# legitimately-occluded real person (legs under a desk → inferred) still clears
# it. HONEST LIMIT (per the 2026-07-15 adversarial audit): this does NOT stop a
# true mirror/TV reflection — its joints track cleanly at state 2 — which would
# need a separate spatial/depth-plane heuristic, not a joint-reliability check.
_MIN_REAL_TRACKED_JOINTS = 6
_REAL_CORE_JOINTS = ("spine_shoulder", "spine_mid", "spine_base", "neck", "head")


def _body_is_real(joints: dict) -> bool:
    """True when a parsed joint dict looks like a genuine tracked person — at
    least _MIN_REAL_TRACKED_JOINTS fully-tracked joints AND a fully-tracked core
    joint — rather than an inferred/zero-fill ghost. Pure; never raises (an
    unusable dict reads as NOT real, failing safe toward "no phantom")."""
    try:
        if not joints:
            return False
        reliable = 0
        for j in joints.values():
            if _joint_reliable(j):
                reliable += 1
        if reliable < _MIN_REAL_TRACKED_JOINTS:
            return False
        return any(_joint_reliable(joints.get(n)) for n in _REAL_CORE_JOINTS)
    except Exception:   # pragma: no cover - defensive: malformed joints dict
        return False


def _body_facing_yaw(joints: dict) -> Optional[float]:
    """Estimate the body's facing YAW in degrees from skeleton JOINT POSITIONS,
    or None when the joints needed aren't tracked.

    WHY POSITIONAL (not the Face API): this pykinect2 build exposes NO Kinect v2
    Face API — there is no IFaceFrameSource / IHighDefinitionFaceFrameSource, no
    FaceFrameFeatures_RotationOrientation, and PyKinectRuntime wires no face
    reader (verified live: those symbols are absent). So we recover facing from
    the geometry the body stream DOES give us reliably: the shoulder line.

    GEOMETRY: in Kinect camera space x points to the sensor's right, z points
    away from the sensor (depth, metres), y points up. The vector from the LEFT
    shoulder to the RIGHT shoulder lies along the chest. When the user squarely
    faces the sensor that vector runs along +x at constant depth (dz≈0). When
    they rotate to look at a side monitor, the shoulder they turn toward moves
    closer in z, so dz grows. The facing direction is the shoulder line rotated
    -90° about the vertical, which works out to:

        yaw = atan2(dz_LR, dx_LR)      # dx = xR - xL, dz = zR - zL

    yielding 0° when squarely facing the sensor, NEGATIVE when the user turns to
    THEIR right / the sensor's left (a left-hand monitor), POSITIVE when they
    turn to THEIR left / the sensor's right (a right-hand monitor). (Sign chosen
    so it matches a real desk: turning toward a monitor on your left reads
    negative.) A secondary cue — the head's x-offset from the shoulder midpoint —
    nudges the estimate the same direction when shoulders are nearly square but
    the head has already turned, and is averaged in when both shoulders and head
    are well tracked.

    ACCURACY (be honest): this is BODY/shoulder facing, not eyeball gaze. It's a
    coarse signal — roughly ±10-15° once smoothed, and only meaningful while the
    torso actually turns with the head (which is the normal multi-monitor case:
    you swivel your chair / torso toward the screen you work on). A pure
    eyes-only flick with a locked torso will NOT register. It is plenty to tell a
    hard left monitor from a centre from a hard right one; it is NOT a substitute
    for an HD-face gaze vector. Calibration (skills/face_tracker) maps the
    observed yaw band per monitor so the absolute offset of a given desk doesn't
    matter."""
    import math
    sl = joints.get("shoulder_left")
    sr = joints.get("shoulder_right")
    yaw_shoulder: Optional[float] = None
    # _joint_reliable (state>=2, finite, non-zero-fill), NOT _tracked (accepts
    # Inferred state 1): an inferred/occluded shoulder gives a bogus yaw that flows
    # through get_presence().head_yaw_deg → the gaze-monitor picker, naming the
    # wrong screen off a guess. Reliable-only → yaw stays None and the picker keeps
    # the prior monitor rather than swinging on noise. (2026-07-15 ghost audit)
    if _joint_reliable(sl) and _joint_reliable(sr):
        dx = float(sr[0]) - float(sl[0])
        dz = float(sr[2]) - float(sl[2])
        # Degenerate (both shoulders coincident / vertical) → no shoulder yaw.
        if abs(dx) > 1e-4 or abs(dz) > 1e-4:
            yaw_shoulder = math.degrees(math.atan2(dz, dx))

    # Secondary cue: head displaced from the shoulder midpoint along x. Turning
    # to look at a monitor on the sensor's RIGHT shifts the head toward +x of the
    # shoulder centre (→ positive), toward the sensor's LEFT shifts it -x (→
    # negative) — the SAME sign convention as the shoulder term, so the two
    # average cleanly. Scaled to a gentle degrees nudge; only used when we have
    # both a head and a shoulder span to normalise against.
    yaw_head: Optional[float] = None
    head = joints.get("head")
    if _joint_reliable(head) and _joint_reliable(sl) and _joint_reliable(sr):
        mid_x = (float(sl[0]) + float(sr[0])) / 2.0
        span = abs(float(sr[0]) - float(sl[0]))
        if span > 0.05:   # a plausible shoulder width in metres
            # offset in [-1, 1]-ish of half-span; map to ±~35° of head turn.
            offset = (float(head[0]) - mid_x) / (span / 2.0)
            offset = max(-1.5, min(1.5, offset))
            yaw_head = offset * 35.0

    vals = [v for v in (yaw_shoulder, yaw_head) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _parse_body_frame(frame) -> list[dict]:
    """Parse ONE raw PyKinectRuntime body frame into the public list-of-dicts
    shape get_bodies() returns. Pure (no sensor / runtime contact) so the PUMP
    can call it once per frame to populate the shared cache and a unit test can
    feed it a fabricated frame. NEVER raises — a malformed frame degrades to [].

    Each emitted entry:
        {"id": int,
         "joints": {name: (x, y, z, tracking_state), ...},  # metres
         "head": (x, y, z) | None,
         "distance_m": float | None,    # head/spine z
         "facing": bool | None,
         "facing_yaw_deg": float | None,   # 0=square, -=sensor-left, +=sensor-right
         "hand_right": "open"|"closed"|"lasso"|"unknown",
         "hand_left":  "open"|"closed"|"lasso"|"unknown"}"""
    try:
        # NB: frame.bodies on real hardware is a length-6 numpy ndarray
        # (dtype=object) — NEVER apply bool()/`not` to it or numpy raises
        # ValueError("truth value of an array ... is ambiguous"), which the
        # broad except below would swallow, returning [] on every frame and
        # silently killing the entire body-data plane (gestures, presence,
        # head-yaw, hand-states, point/guard). len()/`is None` are safe for
        # both the ndarray and the list test-fakes; the len() is wrapped so a
        # non-sized bodies attr degrades to [] instead of raising.
        bodies = getattr(frame, "bodies", None) if frame is not None else None
        if bodies is None:
            return []
        try:
            if len(bodies) == 0:
                return []
        except TypeError:   # pragma: no cover - non-sized bodies attr (shouldn't happen)
            return []
        out: list[dict] = []
        for i, body in enumerate(bodies):
            if not getattr(body, "is_tracked", False):
                continue
            joints_raw = getattr(body, "joints", None)
            joints: dict[str, tuple] = {}
            if joints_raw is not None:
                for idx in range(_JOINT_COUNT):
                    try:
                        j = joints_raw[idx]
                        pos = j.Position
                        joints[_JOINT_NAMES[idx]] = (
                            float(pos.x), float(pos.y), float(pos.z),
                            int(getattr(j, "TrackingState", 0)),
                        )
                    except Exception:   # pragma: no cover - per-joint read glitch
                        continue
            # GHOST-SKELETON GATE: drop an is_tracked slot that lacks real tracked
            # structure (an inferred/zero-fill phantom — reflection, furniture, a
            # just-departed person) so it never reaches presence/count, the
            # nearest-body pick, gestures, the gaze yaw, or the HUD overlay. This is
            # the SINGLE parse point every consumer reads, so one gate cleans them
            # all. (2026-07-15 ghost-skeleton audit)
            if not _body_is_real(joints):
                continue
            head = joints.get("head")
            out.append({
                # Prefer the Kinect's stable per-person tracking_id (set from
                # body.TrackingId for every tracked body; PyKinectRuntime.py:406)
                # so a body keeps the same 'id' as the person migrates between
                # the fixed 6 slots. Fall back to the enumerate slot index for
                # the list-based test fakes that carry no tracking_id, and guard
                # the falsy default (-1/0/None) so it also degrades to the slot.
                "id": int(getattr(body, "tracking_id", i) or i),
                "joints": joints,
                "head": (head[0], head[1], head[2]) if head else None,
                "distance_m": _joint_distance(joints),
                "facing": _body_is_facing(joints),
                "facing_yaw_deg": _body_facing_yaw(joints),
                # getattr so an older build lacking these attrs degrades to
                # "unknown" rather than KeyError-ing the whole body out.
                "hand_right": _hand_state_name(getattr(body, "hand_right_state", None)),
                "hand_left": _hand_state_name(getattr(body, "hand_left_state", None)),
            })
        return out
    except Exception:   # pragma: no cover - defensive: mid-stream body-frame glitch
        return []


def get_bodies() -> list[dict]:
    """Tracked bodies as a list of dicts (shape per _parse_body_frame). Empty
    list if none tracked or the sensor is unavailable.

    SHARED-CACHE CONTRACT (the fix for gesture/air-mouse starvation)
    ================================================================
    The Kinect v2 body frame is SINGLE-CONSUMER: get_last_body_frame() clears the
    has_new_body_frame() flag, so the FIRST reader each frame consumes it and
    every other reader that tick sees no new frame. With several pollers competing
    (skeleton renderer, the body pump, the ~18 Hz gesture poller, the ~30 Hz
    air-mouse poller) whoever read first won the frame and the rest got [] — the
    pollers starved and gestures/air-mouse never fired.

    The cure: the always-on body PUMP (_read_and_cache_bodies, ~30 Hz) is now the
    SOLE reader of the sensor frame. It parses the bodies ONCE and stores them in
    a module cache with a monotonic timestamp. EVERY consumer (this accessor, and
    thus get_hand_states/get_presence/get_head_yaw and the skeleton overlay) reads
    that shared cache with NO competition — many calls between two sensor frames
    all return the SAME parsed bodies instead of a spurious [].

    Returns the cached bodies when FRESH (cache stamped < BODY_CACHE_FRESH_SEC
    ago). When the cache is stale or empty (no pump running yet, or the pump just
    reset a stale runtime) it falls back to a ONE-SHOT direct read so a lone
    caller with no pump — or the very first call right after enable — still works.
    NEVER raises; a missing sensor / down runtime returns []."""
    cached = _get_cached_bodies()
    if cached is not None:
        return cached
    # Cache cold/stale. If the pump SHOULD be running but its thread died, this is
    # also where we notice (a consumer asking with a cold cache) — restart it so the
    # body pipe self-heals instead of staying permanently blind (H2).
    _ensure_pump_alive()
    # Do a single NON-CONSUMING fallback read so a lone consumer (or the first tick
    # before the pump warms) isn't left blind — but never steal the frame from the
    # pump (H3): with a live pump this serves the cache and leaves the pending frame
    # for the pump; only a genuinely pump-less caller reads the sensor directly.
    return _read_and_cache_bodies(consume=False)


def get_color_space_mapper():
    """Return a callable ``mapper(x, y, z) -> (px, py) | None`` that projects a
    CAMERA-SPACE metre point to a COLOR-SPACE pixel on the 1920×1080 frame, or
    None when the Kinect is off / absent. NEVER raises.

    This is the seam PART A's skeleton overlay uses: audio/kinect_skeleton's
    pure projector is handed this callable and maps every tracked joint with it
    (so the projector itself stays pykinect2-free + unit-testable). All the
    pykinect2 contact lives HERE.

    WHY THE PER-POINT MAPPER (not PyKinectRuntime.body_joints_to_color_space):
    that convenience method allocates ``numpy.ndarray(..., dtype=numpy.object)``
    and ``numpy.object`` was REMOVED in numpy 1.24+, so it raises on this
    machine's modern numpy — the same breakage class the bridge's patch-loader
    fixes elsewhere. We instead build a ``CameraSpacePoint`` per call and invoke
    the runtime's ``_mapper.MapCameraPointToColorSpace`` directly (the ICoordina-
    teMapper COM method), which returns a ``ColorSpacePoint`` with float .x/.y.

    Returns None if the runtime, its ``_mapper``, or the PyKinectV2 module
    aren't available, so the caller cleanly skips the overlay and shows the
    plain frame instead."""
    rt, _ = get_runtime()
    if rt is None:
        return None
    mapper = getattr(rt, "_mapper", None)
    map_fn = getattr(mapper, "MapCameraPointToColorSpace", None) if mapper else None
    if not callable(map_fn):
        return None
    # PyKinectV2 carries the CameraSpacePoint ctypes Structure we must pass by
    # value. import_pykinect2 is idempotent (sys.modules-cached) and does no
    # sensor contact, so this is cheap; bail gracefully if it can't load.
    try:
        pk2, _rt_mod = import_pykinect2()
    except Exception:   # pragma: no cover - loader already proven at open time
        return None
    csp_type = (getattr(pk2, "CameraSpacePoint", None)
                or getattr(pk2, "_CameraSpacePoint", None))
    if csp_type is None:
        return None

    def _mapper(x: float, y: float, z: float):
        try:
            pt = csp_type()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = float(z)
            cs = map_fn(pt)
            return (float(cs.x), float(cs.y))
        except Exception:   # pragma: no cover - per-joint COM/marshalling glitch
            return None

    return _mapper


def _nearest_body(bodies: list[dict]) -> Optional[dict]:
    """The closest tracked body (smallest positive distance_m), or the first
    body when no distance is known, or None for an empty list. The user at the
    desk is the nearest body, so head-yaw/gaze keys off this one."""
    if not bodies:
        return None
    ranked = sorted(
        bodies,
        key=lambda b: (b.get("distance_m")
                       if isinstance(b.get("distance_m"), (int, float))
                       and b.get("distance_m") > 0 else float("inf")))
    return ranked[0]


def get_hand_states() -> dict:
    """Discrete hand grip for the NEAREST tracked body — the safe accessor the
    air-mouse skill reads. NEVER raises; mirrors the joint accessors' graceful-
    sentinel contract. Shape:
        {"right": "open"|"closed"|"lasso"|"unknown",
         "left":  "open"|"closed"|"lasso"|"unknown",
         "tracked": bool,           # was any body in view this call
         "ts": <monotonic>}
    With no sensor / no body in view, returns both hands "unknown" and
    tracked=False (so a missing Kinect degrades to "I can't see your hand" rather
    than a crash). "Nearest" reuses the same distance_m ranking get_presence and
    the gesture/pointing skills use (the shared _nearest_body helper), so the
    air-mouse follows the same body the rest of JARVIS is tracking."""
    base = {"right": "unknown", "left": "unknown",
            "tracked": False, "ts": time.monotonic()}
    try:
        bodies = get_bodies()
    except Exception:   # pragma: no cover - get_bodies already swallows; belt-and-braces
        return base
    nearest = _nearest_body(bodies)
    if nearest is None:
        return base
    return {
        "right": nearest.get("hand_right", "unknown"),
        "left": nearest.get("hand_left", "unknown"),
        "tracked": True,
        "ts": time.monotonic(),
    }


def get_presence() -> dict:
    """Cheap room-presence summary. NEVER raises — any failure degrades to
    'no one present'. Shape:
        {"present": bool, "count": int, "nearest_m": float | None,
         "facing": bool | None, "head_yaw_deg": float | None,
         "ts": <monotonic>}
    `facing` is True if ANY tracked body looks like it's facing the sensor.
    `head_yaw_deg` is the NEAREST body's facing yaw in degrees (the person at
    the desk) — 0≈square to the sensor, negative=turned toward the sensor's
    left, positive=toward the sensor's right; None when it can't be estimated.
    Computed here (off the same body list the count uses) so the gaze poller
    gets yaw without a second body-frame fetch."""
    base = {"present": False, "count": 0, "nearest_m": None,
            "facing": None, "head_yaw_deg": None, "ts": time.monotonic()}
    try:
        bodies = get_bodies()
    except Exception:   # pragma: no cover - get_bodies already swallows; belt-and-braces
        return base
    if not bodies:
        return base
    distances = [b["distance_m"] for b in bodies if b.get("distance_m")]
    facings = [b["facing"] for b in bodies if b.get("facing") is not None]
    nearest = _nearest_body(bodies)
    yaw = nearest.get("facing_yaw_deg") if nearest else None
    return {
        "present": True,
        "count": len(bodies),
        "nearest_m": round(min(distances), 2) if distances else None,
        "facing": (any(facings) if facings else None),
        "head_yaw_deg": (round(float(yaw), 1) if isinstance(yaw, (int, float))
                         else None),
        "ts": time.monotonic(),
    }


def get_head_yaw() -> Optional[float]:
    """The NEAREST tracked body's facing YAW in degrees, or None when there's no
    body / the joints needed aren't tracked / the Kinect is disabled or absent.
    NEVER raises — the canonical "head direction" accessor the gaze layer reads.

    Convention: ~0° squarely facing the sensor, NEGATIVE when the user has
    turned toward the sensor's LEFT (a left-hand monitor), POSITIVE toward the
    sensor's RIGHT (a right-hand monitor). This is BODY/shoulder-derived facing
    (see _body_facing_yaw for the geometry + honest accuracy notes), NOT an
    HD-face gaze vector — the Kinect v2 Face API is not available on this
    pykinect2 build."""
    try:
        bodies = get_bodies()
    except Exception:   # pragma: no cover - get_bodies already swallows
        return None
    nearest = _nearest_body(bodies)
    if nearest is None:
        return None
    yaw = nearest.get("facing_yaw_deg")
    return float(yaw) if isinstance(yaw, (int, float)) else None


# ─── PART B: stale-runtime guard + always-on body-frame pump ────────────────
# TWO BUGS this pump cures:
#   (1) The runtime streams briefly at boot then the BODY frame stream goes QUIET
#       for minutes while the owner waves: the Kinect body pipe only flows while
#       something READS it, so once reads stop landing has_new_body_frame() stays
#       False forever and every body consumer goes dead though the sensor is fine.
#   (2) The body frame is SINGLE-CONSUMER: get_last_body_frame() clears the
#       new-frame flag, so when several pollers compete (skeleton renderer, this
#       pump, the ~18 Hz gesture poller, the ~30 Hz air-mouse poller) the FIRST
#       reader each frame consumes it and the rest get [] — the pollers starve
#       and gestures/air-mouse never fire even with body data clearly flowing.
#
# THE FIX (shared cache + two layers):
#   1. The ALWAYS-ON pump (started when KINECT_ENABLED) is now the SOLE reader of
#      the sensor frame. Each tick it reads at most one pending frame, parses the
#      bodies ONCE, and publishes them to the shared _body_cache with a monotonic
#      stamp. Run at ~30 Hz so the cache is at most ~33 ms stale — fresh enough
#      for the 30 Hz air-mouse. EVERY consumer (get_bodies + the accessors on it,
#      the skeleton overlay) reads that cache with NO competition, so many reads
#      between two sensor frames all return the SAME bodies (never a spurious []).
#      Singleton-guarded (never two) with a stop event; opens the runtime itself
#      so the cache warms even before any consumer calls in.
#   2. A STALENESS reset: whenever a real new body frame is observed we stamp a
#      monotonic timestamp; if the runtime is open yet no new body frame has
#      arrived for > BODY_STALE_RESET_SEC, reset _runtime[0]=None so the next
#      get_runtime() RE-OPENS the sensor (re-binding a live stream). On builds
#      whose open path verifies streaming before caching, that reopen also
#      retries until frames actually arrive. The pump drives this check, and
#      _read_and_cache_bodies() stamps freshness so a healthy stream never trips it.
BODY_PUMP_INTERVAL_SEC = 1.0 / 30.0   # ~30 Hz: cache ≤~33 ms stale (feeds air-mouse)
BODY_STALE_RESET_SEC = 4.0         # no new body frame for this long → reset+reopen
_last_body_frame_at = [0.0]        # monotonic of the last OBSERVED new body frame
# Color is a FIRST-CLASS stream for the reset decision (the fix for "the skeleton
# preview stopped after a while / on standing up"): a body-only dropout (the owner
# leaving the frustum) must NOT tear down a runtime whose COLOR is still flowing,
# or the preview (which is backed by color) goes dark while gestures keep working.
# get_color_bgr stamps this whenever it delivers a real frame; the stale-reset only
# fires when BOTH body AND color are quiet.
_last_color_frame_at = [0.0]       # monotonic of the last DELIVERED color frame
_body_pump_thread: list[Any] = [None]
_body_pump_stop = threading.Event()
_body_pump_lock = threading.Lock()


def _pump_is_alive() -> bool:
    """True iff the body pump thread exists and is running. A cheap read used by
    the non-consuming fallback (H3) to decide whether a consumer may read the
    sensor directly (no pump → yes) or must leave the frame for the pump (pump
    alive → serve the cache). NEVER raises."""
    t = _body_pump_thread[0]
    try:
        return t is not None and t.is_alive()
    except Exception:   # pragma: no cover - defensive: odd thread object
        return False


def _ensure_pump_alive() -> bool:
    """Restart the always-on body pump if it SHOULD be running (KINECT enabled) but
    its thread is dead or missing — the self-heal for "the pump thread died and
    nothing restarted it, so the sensor went permanently blind" (H2). reset_if_body_
    stale + the reopen run ONLY from the pump, so a dead pump means no stale-reset
    ever fires again; this is called from get_bodies()'s cold/stale fallback,
    available(), and reset_if_body_stale's re-seed so any of those notices + revives
    it. start_body_pump() is singleton-guarded (no-ops when a pump is already alive
    or when disabled), so this is safe to call freely. Returns True iff it (re)started
    a pump. NEVER raises."""
    if not _ENABLED:
        return False
    if _pump_is_alive():
        return False
    try:
        return start_body_pump()
    except Exception:   # pragma: no cover - defensive: start already guards
        return False


def note_body_frame_seen(now: Optional[float] = None) -> None:
    """Stamp the time a fresh body frame was observed (the staleness clock).
    Called by get_bodies() on a real frame AND by the pump, so a healthy stream
    keeps the clock current and never trips the stale-reset."""
    _last_body_frame_at[0] = time.monotonic() if now is None else now


def note_color_frame_seen(now: Optional[float] = None) -> None:
    """Stamp the time a real color frame was delivered (the COLOR staleness clock).
    Called by get_color_bgr on every frame it returns AND by the color-priming pump
    tick, so a healthy color stream keeps this current and the stale-reset can tell
    "body quiet but color live" (a body dropout) from "the whole runtime is dead"."""
    _last_color_frame_at[0] = time.monotonic() if now is None else now


def _color_frame_is_stale(now: Optional[float] = None) -> bool:
    """True iff the runtime is OPEN but no color frame has been delivered for longer
    than BODY_STALE_RESET_SEC. Mirrors _body_frame_is_stale for the color stream so
    the reset can require BOTH planes to be quiet. False when no runtime is open or
    no color frame has yet been seen (the open path seeds the window)."""
    if _runtime[0] is None:
        return False
    last = _last_color_frame_at[0]
    if last <= 0.0:
        return False
    now = time.monotonic() if now is None else now
    return (now - last) > BODY_STALE_RESET_SEC


def _prime_color_frame() -> None:
    """Pull the latest color frame (require_new=False) so the COLOR buffer stays
    WARM independent of the face-track read — the durability net for the preview.
    With this, color survives even if the face loop pauses, and a freshly-reopened
    runtime gets color re-primed immediately instead of staying cold (which is what
    made get_color_bgr(require_new=False) keep returning None after a reset). Cheap:
    a buffer copy when a frame exists, a no-op when not. NEVER raises."""
    try:
        get_color_bgr(require_new=False)
    except Exception:   # pragma: no cover - get_color_bgr already swallows
        pass


def _body_frame_is_stale(now: Optional[float] = None) -> bool:
    """True iff the runtime is OPEN but no new body frame has been observed for
    longer than BODY_STALE_RESET_SEC. Pure (clock + the two module cells) so it
    is unit-testable. False when no runtime is open (nothing to reset) or when a
    frame has never yet been seen on a freshly-opened runtime within the window
    (the timestamp is seeded at open time by reset_stale_runtime's callers /
    available())."""
    if _runtime[0] is None:
        return False
    last = _last_body_frame_at[0]
    if last <= 0.0:
        return False
    now = time.monotonic() if now is None else now
    return (now - last) > BODY_STALE_RESET_SEC


def reset_if_body_stale(now: Optional[float] = None) -> bool:
    """If BOTH the body AND the color streams have gone stale, drop the cached
    runtime so the NEXT get_runtime() RE-OPENS the sensor (re-binding a live
    stream; on a stream-verifying open path it also retries until frames arrive).
    Returns True iff it performed a reset. Holds the same _lock the open/close path
    uses so it can't race a concurrent reopen. NEVER raises.

    COLOR IS NOW PART OF THE DECISION (the preview-keep-alive fix). The old code
    reset on a BODY-only signal, which tore down the whole shared runtime — and the
    color reader thread with it — whenever the tracked body briefly left the frustum
    (e.g. the owner standing up + sitting back down). Gestures recovered (the body
    pump re-primed body) but the preview's color often did not, so the skeleton
    froze then went dark. Requiring BOTH planes quiet means a body dropout whose
    color is still flowing NO LONGER kills the runtime, so the preview keeps
    rendering; a genuinely dead sensor (no body AND no color) still resets."""
    with _lock:
        body_stale = _body_frame_is_stale(now)
        color_stale = _color_frame_is_stale(now)
        if not (body_stale and color_stale):
            return False
        rt = _runtime[0]
        _runtime[0] = None
        # Re-seed BOTH clocks so the freshly-reopened runtime gets a full window to
        # start streaming before it could be judged stale again.
        stamp = time.monotonic() if now is None else now
        _last_body_frame_at[0] = stamp
        _last_color_frame_at[0] = stamp
        # Zero the per-stream seen-cells: the dead instance's frame-time stamps
        # must not carry over to the reopened one (_publish_runtime re-seeds them
        # from the fresh instance on the next successful open).
        _color_time_seen[0] = 0.0
        _body_time_seen[0] = 0.0
        print("  [kinect] body AND color streams stale > "
              f"{BODY_STALE_RESET_SEC:.0f}s - resetting runtime to reopen a live "
              "stream")
    # We already nulled the cached cell under the lock; explicitly release the
    # old handle (outside the lock) so the sensor frees promptly before the next
    # get_runtime() reopens it. Same best-effort close idiom close() uses; older
    # builds without .close() rely on __del__.
    if rt is not None:
        closer = getattr(rt, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:   # pragma: no cover - defensive: close on a half-dead runtime
                pass
    # The reopen is driven by the pump; if the pump thread has died, the reset we
    # just performed would never be followed by a reopen and the sensor would stay
    # dark. Make sure the pump is alive so the next tick reopens the live stream (L2).
    _ensure_pump_alive()
    return True


def _pump_tick() -> None:
    """One body-pump tick — the SOLE reader of the single-consumer body frame.
    Reads any pending frame ONCE via _read_and_cache_bodies(), which parses the
    bodies, publishes them to the shared cache, and stamps freshness; then resets
    the runtime if the stream has gone stale. Every consumer (get_bodies and the
    accessors built on it, the skeleton overlay) then reads that shared cache with
    NO per-frame competition — the cure for the gesture/air-mouse starvation.

    Because the pump now OWNS the read, it also OPENS the runtime when KINECT is
    enabled (via get_runtime() inside _read_and_cache_bodies) so the cache stays
    warm even when no consumer has called in yet. NEVER raises — a glitch just
    skips this tick. Factored out of the loop so it is unit-testable without a
    thread."""
    if not _ENABLED:
        return
    try:
        # Sole frame read → parse → publish to the shared cache (+ stamp the
        # staleness clock when a real frame arrived). This also lazily opens the
        # runtime when enabled, so the body pipe is kept warm with no consumer.
        _read_and_cache_bodies()
    except Exception:   # pragma: no cover - defensive: _read_and_cache_bodies already swallows
        pass
    # Keep the COLOR buffer warm too (preview-keep-alive fix): pull the latest
    # color frame so it stays primed independent of the face-track read, and so a
    # reopened runtime gets color re-primed immediately rather than staying cold
    # (which is what made get_color_bgr(require_new=False) keep returning None).
    _prime_color_frame()
    # Layer 2: reset a runtime ONLY when BOTH body AND color have gone quiet (a
    # body-only dropout whose color is fine must not tear down the preview).
    try:
        reset_if_body_stale()
    except Exception:   # pragma: no cover - reset already swallows
        pass


# PER-LOOP stop Event (M3). The old design shared one module-level Event that a
# new start_body_pump() CLEARED — so a fast set_enabled(False)→(True) could clear
# the stop before the OLD loop noticed it, leaving two pumps fighting over the
# single-consumer body frame. Each loop now closes over its OWN Event and also
# checks its thread identity, so a superseded loop exits promptly even if a newer
# start re-cleared the shared event. _body_pump_stop_current holds the live loop's
# Event so stop_body_pump() can signal exactly the running one; _body_pump_token
# holds its unique identity for the supersede check.
_body_pump_stop_current: list[Any] = [None]
_body_pump_token: list[Any] = [None]


def _body_pump_loop(stop: "threading.Event", token: object) -> None:  # pragma: no cover - non-terminating daemon; _pump_tick is unit-tested directly
    """Always-on daemon: tick the body pump until told to exit. `stop` is THIS
    loop's own Event (not the shared one) and `token` is this loop's unique
    identity; the loop exits when its own stop is set, the shared stop is set, the
    bridge is disabled, OR it has been SUPERSEDED (a newer start installed a
    different token in _body_pump_token) — the identity check that guarantees a fast
    disable→enable can't leave two pumps fighting the single-consumer body frame
    (M3). Cheap — a few getattr + a readiness poll per tick."""
    while not stop.is_set() and not _body_pump_stop.is_set():
        # Superseded by a newer pump, or disabled → stand down so only the current
        # pump reads the single-consumer body frame.
        if _body_pump_token[0] is not token or not _ENABLED:
            return
        try:
            _pump_tick()
        except Exception:
            pass
        stop.wait(BODY_PUMP_INTERVAL_SEC)


def start_body_pump() -> bool:
    """Start the always-on body-frame pump (singleton — never two). Called from
    set_enabled(True). Returns True iff it started a NEW thread; False if one was
    already running or the bridge is disabled. The thread self-exits when its own
    stop Event is set (close/disable) or it is superseded, and is a daemon so it
    never blocks shutdown."""
    with _body_pump_lock:
        if not _ENABLED:
            return False
        t = _body_pump_thread[0]
        if t is not None and getattr(t, "is_alive", lambda: False)():
            return False
        # Fresh PER-LOOP stop Event + identity token so a prior loop (whose Event we
        # do NOT clear, and whose token we now supersede) stands down — the M3 fix.
        # Clear the shared event too (legacy callers / belt-and-braces) without
        # disturbing the old loop, which keys off its own Event + the token check.
        stop = threading.Event()
        token = object()
        _body_pump_stop_current[0] = stop
        _body_pump_token[0] = token
        _body_pump_stop.clear()
        # Seed BOTH freshness clocks so a just-enabled pump gives the streams a
        # full window to come up before the stale-reset could fire.
        now0 = time.monotonic()
        _last_body_frame_at[0] = now0
        _last_color_frame_at[0] = now0
        t = threading.Thread(target=_body_pump_loop, name="kinect-body-pump",
                             args=(stop, token), daemon=True)
        _body_pump_thread[0] = t
        t.start()
        return True


def stop_body_pump() -> None:
    """Signal the body-frame pump to exit. Idempotent + safe when none ran.
    Called from close()/set_enabled(False). Also drops the shared body cache so a
    closed/disabled sensor reports [] immediately rather than serving a stale body
    list for up to BODY_CACHE_FRESH_SEC after the stream is gone."""
    # Signal BOTH the shared event (legacy) AND the live loop's own Event (M3) so
    # the currently-running pump exits even though each loop now keys off its own.
    _body_pump_stop.set()
    cur = _body_pump_stop_current[0]
    if cur is not None:
        try:
            cur.set()
        except Exception:   # pragma: no cover - defensive: odd event object
            pass
    with _body_pump_lock:
        _body_pump_thread[0] = None
        _body_pump_stop_current[0] = None
        _body_pump_token[0] = None
    # Invalidate the cache: with no pump reading, a lingering entry must not be
    # served. None (not []) so get_bodies() treats it as cold → one-shot read.
    with _body_cache_lock:
        _body_cache[0] = None
        _body_cache_at[0] = 0.0


# ─── lifecycle ────────────────────────────────────────────────────────────

def close(final: bool = False) -> None:
    """Release the runtime. Safe to call repeatedly (idempotent) and safe
    when no runtime was ever opened. Also signals the body-frame pump to stop
    (PART B) so a closed sensor has no pump ticking against a dead runtime.

    final=True makes the close STICK: it clears _ENABLED first so nothing can
    re-open the sensor behind us. Without it, close() was RESURRECTABLE — the
    body pump ticks at 30 Hz, so a tick already past its stop-check calls
    get_runtime(), finds _runtime[0] None but _ENABLED still True, opens a
    FRESH PyKinectRuntime, and _publish_runtime even starts a NEW pump thread.
    On the exit path that means a thread is holding a live Kinect driver handle
    microseconds before TerminateProcess — precisely the "driver-parked thread
    can't be reaped" corpse class v2.0.57 exists to prevent (and a plausible
    cause of the sensor staying wedged across restarts). 2026-07-14 audit."""
    global _ENABLED
    if final:
        _ENABLED = False
    stop_body_pump()
    with _lock:
        rt = _runtime[0]
        _runtime[0] = None
        if rt is None:
            return
        # PyKinectRuntime exposes .close() in recent builds; older ones rely on
        # __del__. Try the explicit close, swallow anything.
        closer = getattr(rt, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:   # pragma: no cover - defensive: close on a half-dead runtime
                pass


# ─── drop-in cv2.VideoCapture shim ────────────────────────────────────────

class KinectCapture:
    """A cv2.VideoCapture work-alike backed by the Kinect color stream, so the
    monolith's _open_capture() can hand the face-tracking loop a Kinect source
    without any other change: `.read()` returns (ret, bgr_frame) and
    `.release()` is a no-op-safe teardown.

    The underlying runtime is the shared singleton (opening a Kinect twice
    fails), so .release() does NOT close it — other consumers (presence poller,
    voice actions) may still need it. Call kinect_bridge.close() to actually
    release the sensor."""

    def __init__(self):
        # Touch the runtime so a misconfigured / disabled Kinect surfaces at
        # open time the way cv2.VideoCapture(idx).isOpened() would.
        rt, err = get_runtime()
        self._opened = rt is not None
        self._open_error = err

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        """Return (ret, frame) like cv2.VideoCapture.read(): (True, bgr) with
        the most recent color frame, or (False, None) only when the sensor is
        genuinely unavailable. require_new=False so a poll faster than the
        sensor's frame rate returns the last frame rather than a false
        read-failure that would make the face-track loop reopen a webcam."""
        frame = get_color_bgr(require_new=False)
        if frame is None:
            return False, None
        return True, frame

    def set(self, *_a, **_k) -> bool:
        # cv2 callers set FRAME_WIDTH/HEIGHT/BUFFERSIZE; the Kinect resolution
        # is fixed, so accept and ignore (matches cv2 returning False for an
        # unsupported prop without raising).
        return False

    def get(self, prop):
        # Report the fixed Kinect color geometry for the two props the
        # face-track open path reads back (CAP_PROP_FRAME_WIDTH=3, HEIGHT=4).
        if prop == 3:
            return 1920.0
        if prop == 4:
            return 1080.0
        return 0.0

    def release(self) -> None:
        # Do NOT close the shared singleton here (see class docstring).
        self._opened = False
