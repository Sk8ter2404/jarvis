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

SPEED PLAN R5 (2026-10-02): THE CALLERS TAGGED LATER
====================================================
The first tags (learn_from_turn, the ambient extractor / judge / screen
observer, the session checkpoint, the LTM reflector) wait
for real. The background callers found untagged afterwards (the notification
classifier, Chappie's daemon, the credits monitor, the scheduled evening
briefing, the morning chain's briefing / handoff) are tagged
``background_work(tag, opt_in=True)``: such a job waits
only while the gate's ``strict`` switch is on (the monolith wires it to
BACKGROUND_TAG_STRICT). With the switch off it runs at once, exactly as it did
untagged, and logs ONCE per job that it would have waited (shadow), so the
switch can be judged from the log before it is flipped. ``max_defer_s`` gives
one job a shorter soft cap than the gate's (a notification should not sit out
two minutes of conversation); the hard grace still bounds it.

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

    __slots__ = ("tag", "started_at", "cancel", "opt_in", "max_defer_s",
                 "shadow_logged")

    def __init__(self, tag: str, started_at: float, cancel=None, *,
                 opt_in: bool = False, max_defer_s=None):
        self.tag = tag
        self.started_at = started_at
        self.cancel = cancel
        self.opt_in = bool(opt_in)
        self.max_defer_s = _clean_cap(max_defer_s)
        self.shadow_logged = False


_tls = threading.local()


def _clean_tag(tag) -> str:
    """One whitespace-free token, so a tag stays one key=value field in a log."""
    try:
        t = "_".join(str(tag or "").split())
    except Exception:
        t = ""
    return t[:48] or "background"


def _clean_cap(v):
    """A job's own soft cap: a positive float, or None (use the gate's).
    Anything else -- zero, negative, junk -- is None. Never raises."""
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        return f if f > 0 else None
    except Exception:
        return None


@contextlib.contextmanager
def background_work(tag: str = "background", cancel=None, clock=None, *,
                    opt_in: bool = False, max_defer_s=None):
    """Mark the calling thread as doing NON-URGENT background local-model work.

    `tag` names the job in log lines (a caller name, never user text).
    `cancel` is an optional zero-arg callable; when it returns True a waiting
    job stops waiting and runs at once (a daemon being stopped should not sit
    out the whole deferral). Nested use keeps the OUTER job, so one job's
    total wait is bounded once, from when it first started waiting.
    `opt_in` (speed plan R5): the job waits only while the gate's strict
    switch is on; otherwise it runs at once and logs, once, that it would
    have waited. `max_defer_s`: this job's own soft cap when shorter than the
    gate's (None = the gate's)."""
    outer = getattr(_tls, "job", None)
    if outer is not None:
        yield outer
        return
    job = Job(_clean_tag(tag), (clock or time.monotonic)(), cancel,
              opt_in=opt_in, max_defer_s=max_defer_s)
    _tls.job = job
    try:
        yield job
    finally:
        _tls.job = None


def current_job():
    """The calling thread's background Job, or None (owner / untagged)."""
    return getattr(_tls, "job", None)


# ── who is calling the brain (caller tags, 2026-10-15) ────────────────────
# EVERY call into the brain -- the local model through Ollama, or Claude
# through core.llm_client -- declares WHO is calling (`caller=`, a stable
# snake_case name) and, when it is not the default, whether the call is
# BACKGROUND (`background=True`: learning, summaries, nightly jobs, proactive
# remarks, audits...) rather than FOREGROUND (answering the owner, or doing
# what he just asked). There is no second mechanism: a background call is a
# `background_work` job. The client layer wraps its work in `call_scope`,
# which enters `background_work(caller, opt_in=True)` when the call is
# background and the thread has no job yet, so a declared-background call
# reaches the SAME gate (`slot`) and the same shadow / strict switch
# (BACKGROUND_TAG_STRICT) the hand-wrapped jobs use; a thread that is already
# inside a `background_work` job is background without saying so, and keeps
# its outer tag as the fallback caller name. tests/test_brain_caller_tags.py
# fails any brain call that omits `caller=`.
UNTAGGED = "untagged"


def clean_caller(caller) -> str:
    """One whitespace-free token (same rule as a job tag) or UNTAGGED."""
    try:
        t = "_".join(str(caller or "").split())
    except Exception:
        t = ""
    return t[:48] or UNTAGGED


def resolve_caller(caller="", background=None) -> tuple:
    """(caller name, is_background) for one brain call.

    `background` None means "ask the thread": background when the calling
    thread is inside a background_work job. An explicit True / False wins. The
    name is `caller`, else the enclosing job's tag, else UNTAGGED. Never
    raises."""
    try:
        job = current_job()
        name = clean_caller(caller) if caller else (
            clean_caller(job.tag) if job is not None else UNTAGGED)
        bg = (job is not None) if background is None else bool(background)
        return name, bg
    except Exception:
        return UNTAGGED, False


@contextlib.contextmanager
def call_scope(caller="", background=None):
    """Wrap ONE brain call: yields (caller name, is_background). A background
    call from a thread with no job runs as background_work(name, opt_in=True),
    so it passes the gate exactly like every other tagged job (it waits only
    while BACKGROUND_TAG_STRICT is on; otherwise it runs at once and the gate
    logs once that it would have waited). Foreground calls, and calls already
    inside a job, are untouched."""
    name, bg = resolve_caller(caller, background)
    if bg and current_job() is None:
        with background_work(name, opt_in=True):
            yield name, True
        return
    yield name, bg


class CallLedger:
    """Per-caller totals of this process's brain calls, so a later report can
    show cost per caller and the foreground / background split. Counts only
    (never prompt or reply text). Bounded: past MAX_CALLERS new names fold into
    'other'. Thread-safe; never raises."""

    MAX_CALLERS = 96
    FIELDS = ("calls", "input", "output", "cache_read", "cache_write")

    def __init__(self):
        self._lock = threading.Lock()
        self._rows: dict = {}

    @staticmethod
    def _n(v) -> int:
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0

    def note(self, caller, background, route, usage=None) -> None:
        """One finished call. `route` is 'cloud' or 'local'; `usage` is an
        optional dict of input / output / cache_read / cache_write tokens."""
        try:
            name = clean_caller(caller)
            bg = bool(background)
            with self._lock:
                key = (name, bg)
                if key not in self._rows and len(self._rows) >= self.MAX_CALLERS:
                    key = ("other", bg)
                row = self._rows.get(key)
                if row is None:
                    row = {"route": {}, **dict.fromkeys(self.FIELDS, 0)}
                    self._rows[key] = row
                row["calls"] += 1
                r = route if route in ("cloud", "local") else "other"
                row["route"][r] = row["route"].get(r, 0) + 1
                for f in self.FIELDS[1:]:
                    row[f] += self._n((usage or {}).get(f))
        except Exception:
            pass

    def snapshot(self) -> dict:
        """{caller: {"background": bool, "calls", "input", ... , "route": {..}}}.
        A name seen both ways keeps one row per side, the background one under
        '<name>' + ' [bg]'."""
        with self._lock:
            out = {}
            for (name, bg), row in self._rows.items():
                key = f"{name} [bg]" if bg and (name, False) in self._rows else name
                out[key] = {"background": bg, **{k: (dict(v) if isinstance(v, dict) else v)
                                                 for k, v in row.items()}}
            return out

    def totals(self) -> dict:
        """{"foreground": {...}, "background": {...}} summed over callers."""
        out = {"foreground": dict.fromkeys(self.FIELDS, 0),
               "background": dict.fromkeys(self.FIELDS, 0)}
        with self._lock:
            for (_name, bg), row in self._rows.items():
                side = out["background" if bg else "foreground"]
                for f in self.FIELDS:
                    side[f] += row[f]
        return out

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()


CALLS = CallLedger()


def note_call(caller, background, route, usage=None) -> None:
    """Record one finished brain call in the process-wide ledger."""
    CALLS.note(caller, background, route, usage)


def caller_label(caller, background) -> str:
    """'caller=<name> fg|bg' for a log line. Never raises."""
    try:
        return f"caller={clean_caller(caller)} {'bg' if background else 'fg'}"
    except Exception:
        return "caller=? fg"


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
    or main thread), 'shadow' (an opt-in job while the strict switch is off:
    it did not wait). `holds` says whether release() must free the slot."""

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
    acquire returns at once and nothing is serialised).
    strict() -> may an opt-in job (background_work(opt_in=True)) wait? None
    (unconfigured) = yes; a raising predicate = no."""

    def __init__(self, defer_reason=None, max_defer_s=None, *,
                 hard_grace_s: float = DEFAULT_HARD_GRACE_S,
                 poll_s: float = DEFAULT_POLL_S, clock=None, log=None,
                 strict=None):
        self._cond = threading.Condition(threading.Lock())
        self._defer_reason = defer_reason
        self._max_defer_s = max_defer_s
        self._strict_src = strict
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
                  hard_grace_s=None, poll_s=None, strict=None) -> None:
        with self._cond:
            if defer_reason is not None:
                self._defer_reason = defer_reason
            if max_defer_s is not None:
                self._max_defer_s = max_defer_s
            if strict is not None:
                self._strict_src = strict
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

    def _job_cap(self, max_defer_s=None) -> float:
        """The soft cap for one acquire: the gate's, or the job's own when
        that is shorter. A disabled gate (cap <= 0) stays disabled."""
        cap = self._cap()
        own = _clean_cap(max_defer_s)
        if cap > 0 and own is not None:
            return min(cap, own)
        return cap

    def _reason(self):
        try:
            fn = self._defer_reason
            return fn() if fn is not None else None
        except Exception:
            return None     # a broken predicate must never hold a job back

    def strict(self) -> bool:
        """May an opt-in job wait? Unconfigured = yes (it then behaves like
        every other tagged job); a raising predicate = no, because a broken
        switch must never hold a job back. Never raises."""
        try:
            src = self._strict_src
            if src is None:
                return True
            return bool(src() if callable(src) else src)
        except Exception:
            return False

    def note_shadow(self, job):
        """An opt-in `job` is about to run unheld (strict off). Log ONCE per
        job, and only when the gate would have held it right now, that it
        would have waited. Returns the reason it would have waited, or None.
        Never raises."""
        try:
            if self._job_cap(getattr(job, "max_defer_s", None)) <= 0:
                return None             # a disabled gate holds nothing
            reason = self._reason()
            if reason is not None and job is not None and not job.shadow_logged:
                job.shadow_logged = True
                self._say(f"  [bg-local] shadow {job.tag} would defer "
                          f"({reason})")
            return reason
        except Exception:
            return None

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

    def defer_reason(self):
        """Why a background job would wait right now: the configured
        predicate's reason ('conversation', 'turn', 'utterance'), or None.
        For background work that does not go through acquire() (the RAG
        indexer's embeddings, 2026-10-04). Never raises (None)."""
        return self._reason()

    def busy(self) -> bool:
        with self._cond:
            self._drop_dead_holder()
            return self._holder is not None or bool(self._queue)

    def waiting(self) -> int:
        with self._cond:
            return len(self._queue)

    def acquire(self, tag: str = "background", started_at=None,
                cancel=None, max_defer_s=None) -> Pass:
        """Wait (bounded) for the slot. `max_defer_s` is the job's own soft
        cap; it only ever shortens the gate's."""
        tag = _clean_tag(tag)
        if _is_main_thread():
            return Pass("off", tag=tag)
        me = threading.current_thread()
        with self._cond:
            self._drop_dead_holder()
            if self._holder is me:
                self._depth += 1
                return Pass("reentrant", holds=True, tag=tag)
        cap = self._job_cap(max_defer_s)
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
                    cap = self._job_cap(max_defer_s)
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
    thread. Re-entrant per thread. An opt-in job (speed plan R5) while the
    gate's strict switch is off yields Pass('shadow') at once: it never
    waits, never takes the slot, and logs once when it would have waited."""
    g = GATE if gate is None else gate
    job = current_job()
    if job is None or _is_main_thread():
        yield None
        return
    if job.opt_in and not g.strict():
        g.note_shadow(job)
        yield Pass("shadow", tag=job.tag)
        return
    p = g.acquire(job.tag, started_at=job.started_at, cancel=job.cancel,
                  max_defer_s=job.max_defer_s)
    try:
        yield p
    finally:
        g.release(p)


def wait_for_quiet(gate=None) -> str:
    """For a tagged background job about to CAPTURE its input (a screenshot):
    wait until a background request may run, then return at once (the slot
    is taken again by the POST itself). Returns the Pass outcome ('shadow'
    for an opt-in job with the strict switch off), or 'none' for an untagged /
    main-thread caller."""
    with slot(gate) as p:
        return p.outcome if p is not None else "none"
