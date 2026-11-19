"""core/gpu_sampler.py - ONE cached GPU reading per process (v3.0.26).

WHY (overhaul finding reliability-05)
=====================================
The unified HUD started ``nvidia-smi --query-gpu=utilization.gpu`` once a
second for as long as it ran (TICK_MS 500, GPU_CACHE_SECONDS 1.0: ~86,000
starts a day), on top of every other GPU poller, each with its own
nvidia-smi: core/gpu_usage (the HUD's VRAM panel, ~6 s), skills/system_pulse
(the strip, 15 s, one spawn for utilization and one for temperature),
skills/status_panel (20 s), core/running_costs (power draw), and the two
on-demand overlays (holographic HUD v2, arc reactor status). Every start
loads the NVIDIA management library and initialises it on both cards - a
burst of CPU and driver work, all day, while the owner games.

core/gpu_probe already reads that library (NVML) in-process: initialised
once per process, no CUDA context, under a millisecond a read. This module
is the one sampler those pollers share: ``sample(max_age_s)`` returns the
cached reading while it is fresh enough, else reads NVML once and caches
it. The read runs on a short-lived daemon thread (at most one in flight) and
a caller waits at most _READ_WAIT_S for it, so a wedged driver call can
never hold the voice thread; the caller then gets the last reading. Nothing
here spawns a process.

NVML ONLY, ON PURPOSE: with no NVML (no NVIDIA driver, or a test process,
where tools/hermetic_guard refuses the library like it refuses nvidia-smi)
``sample`` returns None and each caller keeps its own nvidia-smi fallback,
exactly as before - so a box without the library loses nothing, and this
box, which has it, spawns nothing.

Public API (stdlib + core.gpu_probe, never raises):
    sample(max_age_s=DEFAULT_MAX_AGE_S) -> dict | None
        {"util_pct", "temp_c", "used_mb", "free_mb", "total_mb", "power_w",
         "gpus": [per-card dicts from gpu_probe.gpus()], "source": "nvml",
         "ts": monotonic seconds}; aggregate values are None when no card
        reported them. used_mb is what nvidia-smi calls memory.used (the
        allocated memory, without the driver's reservation) when the driver
        can say, else NVML v1's used.
    util_pct(max_age_s=...) -> float | None     busiest card's utilization
    temp_c(max_age_s=...) -> float | None       hottest card's temperature
"""
from __future__ import annotations

import threading
import time

from core import gpu_probe

DEFAULT_MAX_AGE_S = 2.0
# The NVML read runs on its own short-lived daemon thread and a caller waits
# at most this long for it, then takes the cached value (stale or None)
# instead: an NVML read is well under a millisecond, and this bound is for a
# wedged driver call, which must never hold a caller (the voice thread asks
# through the pulse / panel / VRAM replies). At most ONE read is in flight;
# while one is stuck, no other starts.
_READ_WAIT_S = 0.5
_LOCK_WAIT_S = 0.5      # _lock guards only dict updates; never held across a read

_lock = threading.Lock()
# at = monotonic stamp; gen = gpu_probe's loader generation it was read with
# (a set_loader since then makes it stale).
_cache: dict = {"sample": None, "at": None, "gen": None}
_inflight: dict = {"thread": None, "done": None}
_slow_logged = [False]
_clock = time.monotonic                      # tests: set_clock()


def set_clock(fn=None) -> None:
    """Replace the monotonic clock (tests) and forget the cached reading."""
    global _clock
    _clock = fn if fn is not None else time.monotonic
    reset()


def reset() -> None:
    """Forget the cached reading and any in-flight read (tests;
    gpu_probe.set_loader users). A read still running lands harmlessly."""
    _cache["sample"] = None
    _cache["at"] = None
    _cache["gen"] = None
    _inflight["thread"] = None
    _inflight["done"] = None


def _gen():
    try:
        return gpu_probe._state.get("gen", 0)
    except Exception:
        return None


def _fresh(age_ok: float) -> bool:
    at = _cache["at"]
    return (at is not None and _cache["gen"] == _gen()
            and (_clock() - at) < age_ok)


def _max(values):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _sum(values):
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _read() -> "dict | None":
    """One NVML read, aggregated over the readable cards; None when NVML is
    unavailable or reads no card."""
    cards = gpu_probe.gpus()
    if not cards:
        return None
    used = []
    for g in cards:
        alloc = g.get("alloc_mb")
        used.append(alloc if alloc is not None else g.get("used_mb"))
    power = _sum(g.get("power_w") for g in cards)
    return {
        "util_pct": _max(g.get("util_pct") for g in cards),
        "temp_c": _max(g.get("temp_c") for g in cards),
        "used_mb": _sum(used),
        "free_mb": _sum(g.get("free_mb") for g in cards),
        "total_mb": _sum(g.get("total_mb") for g in cards),
        "power_w": round(power, 1) if power is not None else None,
        "gpus": [dict(g) for g in cards],
        "source": "nvml",
        "ts": _clock(),
    }


def _stale() -> "dict | None":
    """The cached reading when it came through the current library, else
    None."""
    return _cache["sample"] if _cache["gen"] == _gen() else None


def _start_read() -> "threading.Event | None":
    """The in-flight read's done-event, starting one when none is running.
    None when the lock could not be taken (then: the cached value)."""
    if not _lock.acquire(timeout=_LOCK_WAIT_S):
        return None
    try:
        t = _inflight["thread"]
        alive = getattr(t, "is_alive", None)
        if (t is not None and callable(alive) and alive()
                and _inflight["done"] is not None):
            return _inflight["done"]
        done = threading.Event()
        gen = _gen()

        def _work():
            snap = None
            try:
                snap = _read()
            except Exception:
                snap = None
            finally:
                if _lock.acquire(timeout=_LOCK_WAIT_S):
                    try:
                        _cache["sample"] = snap
                        _cache["at"] = _clock()
                        _cache["gen"] = gen
                    finally:
                        _lock.release()
                done.set()

        th = threading.Thread(target=_work, name="gpu-sampler", daemon=True)
        _inflight["thread"] = th
        _inflight["done"] = done
        th.start()
        return done
    finally:
        _lock.release()


def sample(max_age_s: float = DEFAULT_MAX_AGE_S) -> "dict | None":
    """The process's GPU reading, at most ``max_age_s`` old (see the module
    docstring). None = no NVML here (the caller's own fallback applies).
    Waits at most _READ_WAIT_S for a new read. Never raises."""
    try:
        age_ok = max(0.0, float(max_age_s))
    except Exception:
        age_ok = DEFAULT_MAX_AGE_S
    try:
        if _fresh(age_ok):
            return _cache["sample"]
        done = _start_read()
        if done is None or not done.wait(_READ_WAIT_S):
            if done is not None and not _slow_logged[0]:
                _slow_logged[0] = True
                print(f"  [gpu-sampler] an NVML read is still running after "
                      f"{_READ_WAIT_S:.1f} s; callers get the last reading "
                      f"until it returns (logged once)", flush=True)
            return _stale()
        return _stale()
    except Exception:
        return None


def util_pct(max_age_s: float = DEFAULT_MAX_AGE_S) -> "float | None":
    """The busiest card's utilization percent, or None. Never raises."""
    try:
        s = sample(max_age_s)
        v = s.get("util_pct") if s else None
        return float(v) if v is not None else None
    except Exception:
        return None


def temp_c(max_age_s: float = DEFAULT_MAX_AGE_S) -> "float | None":
    """The hottest card's temperature in Celsius, or None. Never raises."""
    try:
        s = sample(max_age_s)
        v = s.get("temp_c") if s else None
        return float(v) if v is not None else None
    except Exception:
        return None
