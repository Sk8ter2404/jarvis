"""core/ct2_host.py - ONE long-lived thread for every CTranslate2 call on CUDA.

WHY THIS EXISTS (the v2.0.179 crash, 2026-10-04 20:40)
------------------------------------------------------
CTranslate2 (faster-whisper's engine) keeps CUDA state PER HOST THREAD - a
CUDA stream and cuBLAS / cuDNN handles in C++ ``thread_local`` objects -
created on whatever thread calls into it and destroyed when THAT thread
exits. Their destructors call ``cudaSetDevice`` and throw on failure; a throw
from a destructor is std::terminate -> abort, which takes all of JARVIS down
("Fatal Python error: Aborted"). That is what killed v2.0.179: the
self-diagnostic's throwaway ``probe-stt`` thread decoded one second of tone
on cuda:1, exited, and its CTranslate2 destructor found no CUDA context (the
crash dumps prove the throw site and the driver error; see
core/cuda_preinit.py for why 178 survived the same exit).

core/cuda_preinit.py restores the DLL order that made thread exit survivable.
This module removes the hazard itself: a CTranslate2 call on a CUDA model -
the model load, every decode AND the drain of faster-whisper's lazy segment
generator (where the decode really runs), and the release of a dropped model
- runs on ONE daemon thread that never exits. No thread that ends (a probe,
an ambient worker, a per-turn helper) ever owns CTranslate2 CUDA state, so
thread exit cannot reach those destructors whatever the DLL order or the
driver's teardown rules turn out to be.

CONTRACT
--------
* ``run(fn, *a, **kw)`` runs ``fn`` on the host thread and returns its result
  or re-raises its exception in the caller (frames cleared first, on the host,
  so the traceback does not carry CUDA objects back to a thread that may
  exit). Inline when already on the host (no self-deadlock).
* The wait is UNBOUNDED on purpose: it replaces the caller doing the same
  native call itself, which was unbounded too. A timeout here would only
  abandon a decode that keeps running - and unlike the old in-caller decode,
  an abandoned job still cannot overlap another, because the host runs one
  job at a time.
* ``run_for(device, fn, ...)``: the host for a CUDA device ("cuda",
  "cuda:N"), inline for anything else - CTranslate2 on the CPU has no CUDA
  thread state, and leaving CPU models alone keeps their behaviour unchanged.
* ``decode(model, audio, device, **kw)``: ``model.transcribe(audio, **kw)``
  plus the full drain of its segment generator in ONE host job; returns
  ``(segments_list, info)``.
* ``retire(obj)``: drop the last reference to a CUDA model ON the host. The
  caller hands it over and drops its own references at once; the host keeps
  it for a short grace period (_RETIRE_GRACE_S - long enough for the caller's
  except-block and locals to be gone) and then releases it, so CTranslate2's
  teardown runs on the host. If something else still holds the model then,
  it is freed wherever that last reference goes - exactly as before.

Stdlib only; never imports ctranslate2 itself.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback

_THREAD_NAME = "ct2-host"
_RETIRE_GRACE_S = 2.0         # a retired object is released this long after retire()
_RETIRE_POLL_S = 0.25
_HOST_CHECK_S = 1.0           # how often a waiting caller checks the host is alive

_q: "queue.SimpleQueue" = queue.SimpleQueue()
_thread: list = [None]
_start_lock = threading.Lock()
_graveyard: list = []          # [ [obj, retired_at] ] - touched on the host only
_graveyard_in: "queue.SimpleQueue" = queue.SimpleQueue()
_stats = {"jobs": 0, "retired": 0, "released": 0}


class _Job:
    __slots__ = ("fn", "args", "kwargs", "done", "result", "exc")

    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.done = threading.Event()
        self.result = None
        self.exc = None


def is_cuda(device) -> bool:
    """True for a CUDA device string ("cuda", "cuda:1"). Never raises."""
    try:
        return str(device or "").strip().lower().startswith("cuda")
    except Exception:
        return False


def on_host() -> bool:
    """True when the calling thread IS the host thread."""
    t = _thread[0]
    return t is not None and threading.current_thread() is t


def host_thread():
    """The host thread object (None until the first job)."""
    return _thread[0]


def stats() -> dict:
    """Counters: jobs run, objects retired, objects released on the host."""
    return dict(_stats)


def _drain_graveyard(now=None) -> None:
    """Release retired objects whose grace period is over. Runs on the host
    thread only."""
    while True:
        try:
            _graveyard.append(_graveyard_in.get_nowait())
        except queue.Empty:
            break
    if not _graveyard:
        return
    now = time.monotonic() if now is None else now
    keep = []
    for entry in _graveyard:
        if now - entry[1] >= _RETIRE_GRACE_S:
            entry[0] = None            # the host's reference goes here
            _stats["released"] += 1
        else:
            keep.append(entry)
    _graveyard[:] = keep


def _loop() -> None:  # pragma: no cover - the daemon body; run() is tested through it
    while True:
        try:
            job = _q.get(timeout=_RETIRE_POLL_S if (_graveyard or not
                                                    _graveyard_in.empty())
                         else None)
        except queue.Empty:
            job = None
        try:
            _drain_graveyard()
        except Exception:
            pass
        if job is None:
            continue
        try:
            job.result = job.fn(*job.args, **job.kwargs)
        except BaseException as e:      # handed to the caller, never raised here
            try:
                traceback.clear_frames(e.__traceback__)
            except Exception:
                pass
            job.exc = e
        finally:
            _stats["jobs"] += 1
            job.fn = job.args = job.kwargs = None
            job.done.set()
            job = None
        try:
            _drain_graveyard()
        except Exception:
            pass


def _ensure_started():
    t = _thread[0]
    if t is not None and t.is_alive():
        return t
    with _start_lock:
        t = _thread[0]
        if t is not None and t.is_alive():
            return t
        t = threading.Thread(target=_loop, name=_THREAD_NAME, daemon=True)
        _thread[0] = t
        t.start()
        return t


def run(fn, /, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` on the host thread and return its result
    (or raise its exception). Inline when called from the host itself."""
    if on_host():
        return fn(*args, **kwargs)
    host = _ensure_started()
    job = _Job(fn, args, kwargs)
    _q.put(job)
    while not job.done.wait(_HOST_CHECK_S):
        if not host.is_alive() and not job.done.is_set():
            # Unreachable by design (the loop catches everything per job) -
            # but a caller must never wait forever on a thread that is gone.
            raise RuntimeError("the ct2-host thread is not running")
    exc = job.exc
    if exc is not None:
        job.exc = None
        try:
            raise exc
        finally:
            exc = None
    result = job.result
    job.result = None
    return result


def run_for(device, fn, /, *args, **kwargs):
    """``run`` for a CUDA device, a plain inline call otherwise."""
    if is_cuda(device):
        return run(fn, *args, **kwargs)
    return fn(*args, **kwargs)


def _decode_job(model, audio, kwargs):
    gen, info = model.transcribe(audio, **kwargs)
    try:
        return list(gen), info
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        gen = None


def decode(model, audio, device, /, **kwargs):
    """faster-whisper ``model.transcribe(audio, **kwargs)`` AND the drain of
    its segment generator, in one call on the right thread (the host for a
    CUDA ``device``). Returns ``(segments_list, info)``."""
    return run_for(device, _decode_job, model, audio, kwargs)


def retire(obj) -> None:
    """Hand ``obj`` (a dropped CUDA model) to the host so its last reference
    - and CTranslate2's teardown - goes away on the host thread. The caller
    must drop its own references after this call."""
    if obj is None:
        return
    _stats["retired"] += 1
    _graveyard_in.put([obj, time.monotonic()])
    obj = None
    _ensure_started()
    _q.put(None)               # wake the host so it starts watching it
