"""Local-model traffic control (2026-09-29, r6).

WHY
===
Ollama serves the local brain from ONE slot, and that slot's KV cache holds
only the most recent request. Every owner turn carries a ~12.5k-token prompt
whose front is built to be byte-identical turn to turn, so a turn that merely
extends the previous request costs a few hundred ms of prompt evaluation. But
ANY other local request in between (the per-turn memory extractor, the ambient
extractor, a Teams-nudger screenshot read by the vision model) replaces the
cache, and the next turn pays the full ~2.2 s re-evaluation. Live, that was
every turn.

WHAT THIS MODULE HOLDS (stdlib only, so the CI-light tier covers it)
====================================================================
* ``background_work(tag)`` -- a context manager a NON-URGENT background job
  wraps its local-model work in. It tags the calling thread; nothing else
  changes. Untagged calls (the owner's turn, its follow-ups, an action the
  owner asked for) are never delayed.
* ``BackgroundGate`` / ``GATE`` -- the FIFO the monolith's local POST sites
  pass a tagged thread through (``slot()``). A tagged job waits while the
  owner is in a conversation (the monolith installs that predicate with
  ``GATE.configure``), then runs; jobs run one at a time in arrival order, so
  a burst of queued work never piles up in Ollama's queue in front of the
  owner's next turn. Every wait is bounded: after ``max_defer_s`` the job runs
  anyway (preferring a gap between turns), and after ``max_defer_s +
  hard_grace_s`` it runs no matter what. The main thread never waits.
* ``InferenceTracker`` / ``TRACKER`` -- in-flight count and last-completed
  time of JARVIS's own local inference POSTs (so the system-pulse skill can
  tell its own GPU load from someone else's) and a running POST count (so the
  idle re-prime can tell whether its warm prefix was evicted before the next
  turn used it).

Log lines carry numbers and caller tags only -- never prompt or speech text.
"""
from __future__ import annotations

import contextlib
import threading
import time

# Busy reasons that mean "the owner is mid-sentence or mid-turn right now".
# Past the soft cap a forced job still waits these out (up to the hard grace),
# so it lands BETWEEN turns instead of in front of the owner's next POST.
HARD_REASONS = frozenset({"turn", "utterance"})

DEFAULT_MAX_DEFER_S = 120.0
DEFAULT_HARD_GRACE_S = 30.0
DEFAULT_POLL_S = 0.5


# ── thread tag ────────────────────────────────────────────────────────────
class Job:
    """The background job the current thread is running (see background_work)."""

    __slots__ = ("tag", "started_at", "cancel")

    def __init__(self, tag: str, started_at: float, cancel=None):
        self.tag = tag
        self.started_at = started_at
        self.cancel = cancel


_tls = threading.local()


def _clean_tag(tag) -> str:
    """One whitespace-free token, so a tag stays one key=value field in a log."""
    try:
        t = "_".join(str(tag or "").split())
    except Exception:
        t = ""
    return t[:48] or "background"


@contextlib.contextmanager
def background_work(tag: str = "background", cancel=None, clock=None):
    """Mark the calling thread as doing NON-URGENT background local-model work.

    `tag` names the job in log lines (a caller name, never user text).
    `cancel` is an optional zero-arg callable; when it returns True a waiting
    job stops waiting and runs at once (a daemon being stopped should not sit
    out the whole deferral). Nested use keeps the OUTER job, so one job's
    total wait is bounded once, from when it first started waiting."""
    outer = getattr(_tls, "job", None)
    if outer is not None:
        yield outer
        return
    job = Job(_clean_tag(tag), (clock or time.monotonic)(), cancel)
    _tls.job = job
    try:
        yield job
    finally:
        _tls.job = None


def current_job():
    """The calling thread's background Job, or None (owner / untagged)."""
    return getattr(_tls, "job", None)


def _is_main_thread() -> bool:
    try:
        return threading.current_thread() is threading.main_thread()
    except Exception:
        return False


# ── own-inference tracker ─────────────────────────────────────────────────
class InferenceTracker:
    """In-flight count, last-completed time and a running count of JARVIS's
    own local-model POSTs. Thread-safe; never raises."""

    def __init__(self, clock=None):
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._inflight = 0
        self._last_done = 0.0
        self._posts = 0

    @contextlib.contextmanager
    def track(self, count: bool = True):
        """Wrap ONE local inference POST. `count=False` for a request that
        does not change what the model's cache holds for the next turn (the
        idle re-prime itself)."""
        with self._lock:
            self._inflight += 1
            if count:
                self._posts += 1
        try:
            yield
        finally:
            with self._lock:
                self._inflight = max(0, self._inflight - 1)
                try:
                    self._last_done = float(self._clock())
                except Exception:
                    self._last_done = time.monotonic()

    @property
    def posts(self) -> int:
        with self._lock:
            return self._posts

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def busy_within(self, window_s: float, now=None) -> bool:
        """True while a POST is in flight or one finished < window_s ago."""
        try:
            with self._lock:
                if self._inflight > 0:
                    return True
                last = self._last_done
            if not last:
                return False
            t = float(self._clock() if now is None else now)
            return (t - last) < float(window_s)
        except Exception:
            return False

    def reset(self) -> None:
        with self._lock:
            self._inflight = 0
            self._last_done = 0.0
            self._posts = 0


TRACKER = InferenceTracker()


def own_inference_recent(window_s: float = 10.0) -> bool:
    """Is JARVIS's own local model (chat, vision or re-prime) running now, or
    did it finish within `window_s`? Never raises."""
    try:
        return TRACKER.busy_within(window_s)
    except Exception:
        return False


# ── the background gate ───────────────────────────────────────────────────
class Pass:
    """What BackgroundGate.acquire granted. `outcome` is one of:
    'go' (no wait), 'released' (waited, then the conversation went quiet),
    'forced' (max deferral reached), 'cancelled' (the job's cancel fired),
    'reentrant' (this thread already holds the slot), 'off' (gate disabled
    or main thread). `holds` says whether release() must free the slot."""

    __slots__ = ("outcome", "waited_s", "holds", "tag")

    def __init__(self, outcome: str, waited_s: float = 0.0,
                 holds: bool = False, tag: str = ""):
        self.outcome = outcome
        self.waited_s = waited_s
        self.holds = holds
        self.tag = tag


class BackgroundGate:
    """FIFO slot for non-urgent background local-model requests.

    defer_reason() -> None when a background request may run now, else a
    short reason string ('conversation', or one of HARD_REASONS).
    max_defer_s() -> the soft cap in seconds (<= 0 disables the gate: every
    acquire returns at once and nothing is serialised)."""

    def __init__(self, defer_reason=None, max_defer_s=None, *,
                 hard_grace_s: float = DEFAULT_HARD_GRACE_S,
                 poll_s: float = DEFAULT_POLL_S, clock=None, log=None):
        self._cond = threading.Condition(threading.Lock())
        self._defer_reason = defer_reason
        self._max_defer_s = max_defer_s
        self._hard_grace_s = float(hard_grace_s)
        self._poll_s = float(poll_s)
        self._clock = clock or time.monotonic
        self._log = log
        self._queue: list = []
        self._seq = 0
        # The holding Thread OBJECT, never threading.get_ident(): idents are
        # reused as soon as a thread exits (at once on Linux), so a new thread
        # could pass the holder check and release someone else's slot (CI
        # 2026-09-29, test_quiet_gate_goes_at_once_and_holds_until_release).
        self._holder = None
        self._depth = 0

    def configure(self, *, defer_reason=None, max_defer_s=None,
                  hard_grace_s=None, poll_s=None) -> None:
        with self._cond:
            if defer_reason is not None:
                self._defer_reason = defer_reason
            if max_defer_s is not None:
                self._max_defer_s = max_defer_s
            if hard_grace_s is not None:
                self._hard_grace_s = float(hard_grace_s)
            if poll_s is not None:
                self._poll_s = float(poll_s)

    # -- helpers (never raise) ---------------------------------------------
    def _say(self, msg: str) -> None:
        try:
            (self._log or print)(msg)
        except Exception:
            pass

    def _cap(self) -> float:
        try:
            src = self._max_defer_s
            v = src() if callable(src) else src
            return max(0.0, float(DEFAULT_MAX_DEFER_S if v is None else v))
        except Exception:
            return DEFAULT_MAX_DEFER_S

    def _reason(self):
        try:
            fn = self._defer_reason
            return fn() if fn is not None else None
        except Exception:
            return None     # a broken predicate must never hold a job back

    @staticmethod
    def _cancelled(cancel) -> bool:
        try:
            return bool(cancel()) if cancel is not None else False
        except Exception:
            return False

    # -- public ------------------------------------------------------------
    def _drop_dead_holder(self) -> None:
        """Caller holds _cond. A holder thread that exited without releasing
        (killed daemon, an unexpected BaseException) must not keep the slot:
        clear it so the queue moves instead of every job sitting out the cap."""
        h = self._holder
        if h is not None and not h.is_alive():
            self._holder = None
            self._depth = 0

    def busy(self) -> bool:
        with self._cond:
            self._drop_dead_holder()
            return self._holder is not None or bool(self._queue)

    def waiting(self) -> int:
        with self._cond:
            return len(self._queue)

    def acquire(self, tag: str = "background", started_at=None,
                cancel=None) -> Pass:
        tag = _clean_tag(tag)
        if _is_main_thread():
            return Pass("off", tag=tag)
        me = threading.current_thread()
        with self._cond:
            self._drop_dead_holder()
            if self._holder is me:
                self._depth += 1
                return Pass("reentrant", holds=True, tag=tag)
        cap = self._cap()
        if cap <= 0:
            return Pass("off", tag=tag)
        now = self._clock()
        start = now if started_at is None else min(float(started_at), now)
        outcome = None
        logged = False
        with self._cond:
            ticket = self._seq
            self._seq += 1
            self._queue.append(ticket)
            try:
                while True:
                    now = self._clock()
                    waited = now - start
                    cap = self._cap()
                    reason = self._reason()
                    self._drop_dead_holder()
                    head_free = (self._holder is None
                                 and self._queue and self._queue[0] == ticket)
                    if cap <= 0:
                        outcome = "off"
                        break
                    if head_free and reason is None:
                        outcome = "released" if logged else "go"
                        break
                    if self._cancelled(cancel):
                        outcome = "cancelled"
                        break
                    if waited >= cap + self._hard_grace_s:
                        outcome = "forced"
                        break
                    if waited >= cap and reason not in HARD_REASONS:
                        # Past the soft cap: run now unless the owner is
                        # mid-sentence / mid-turn (then wait for the gap, up
                        # to the hard grace).
                        outcome = "forced"
                        break
                    if not logged:
                        logged = True
                        self._say(f"  [bg-local] defer {tag} "
                                  f"({reason or 'queued'})")
                    limit = (cap if waited < cap
                             else cap + self._hard_grace_s)
                    self._cond.wait(timeout=max(0.01, min(self._poll_s,
                                                          limit - waited)))
            finally:
                try:
                    self._queue.remove(ticket)
                except ValueError:
                    pass
                holds = False
                if outcome is not None and self._holder is None:
                    self._holder = me
                    self._depth = 1
                    holds = True
                self._cond.notify_all()
        waited = max(0.0, self._clock() - start)
        if logged:
            self._say(f"  [bg-local] run {tag} after {int(waited * 1000)} ms "
                      f"({outcome})")
        return Pass(outcome or "forced", waited, holds, tag)

    def release(self, p) -> None:
        if p is None or not getattr(p, "holds", False):
            return
        me = threading.current_thread()
        with self._cond:
            if self._holder is not me:
                return
            self._depth -= 1
            if self._depth <= 0:
                self._holder = None
                self._depth = 0
                self._cond.notify_all()

    def reset(self) -> None:
        """Test / recovery hook: forget every holder and waiter."""
        with self._cond:
            self._queue.clear()
            self._holder = None
            self._depth = 0
            self._cond.notify_all()


# The process-wide gate. The monolith installs the conversation predicate and
# the configured cap at import (GATE.configure); until then defer_reason is
# None, so nothing ever waits (a skill imported without the monolith, tests).
GATE = BackgroundGate()


@contextlib.contextmanager
def slot(gate=None):
    """Hold the background slot for the current thread's background job while
    it POSTs to the local model. A no-op (yields None) for an untagged thread
    -- the owner's turn and anything the owner asked for -- and on the main
    thread. Re-entrant per thread."""
    g = GATE if gate is None else gate
    job = current_job()
    if job is None or _is_main_thread():
        yield None
        return
    p = g.acquire(job.tag, started_at=job.started_at, cancel=job.cancel)
    try:
        yield p
    finally:
        g.release(p)


def wait_for_quiet(gate=None) -> str:
    """For a tagged background job about to CAPTURE its input (a screenshot):
    wait until a background request may run, then return at once (the slot
    is taken again by the POST itself). Returns the Pass outcome, or 'none'
    for an untagged / main-thread caller."""
    with slot(gate) as p:
        return p.outcome if p is not None else "none"
