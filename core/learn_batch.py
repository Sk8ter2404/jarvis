"""Batched ambient learning (2026-10-12).

WHY
===
Ambient / background learning used to make one small local-brain request per
event: one memory extraction per owner turn, one media/person content judge
per overheard line, one multimodal extraction per interval -- about 60 a
session. The local model has ONE slot whose cache holds only the most recent
request, so every one of them evicted the owner's prompt prefix and made the
next owner turn pay a full re-read (core/local_traffic.py explains the cost).
Holding each call back while the owner talks (the background gate) only moves
it; it still lands between two turns, one request at a time.

WHAT THIS MODULE DOES (stdlib only, so the CI-light tier covers it)
===================================================================
A ``LearnBatcher`` collects those small jobs and sends them as ONE call.

* ``submit(job)`` queues a job and returns at once. A job is a prompt
  (``system`` / ``user`` / ``max_tokens``) plus an ``apply(text)`` callback
  that receives that job's own answer. A job may be LAZY (``build()`` returns
  the prompt at flush time, so turns that arrived while it waited ride the
  same call) and then also provides ``unbuild(ctx)`` to hand its input back
  when the batch is not sent.
* A flush happens only when (a) the oldest queued job is at least
  ``interval_s`` old, or ``max_jobs`` are waiting, AND (b) the owner has been
  quiet (``quiet()``: no utterance or turn in progress and no conversation
  activity for a while). Owner turns always win: the quiet check runs before
  the jobs are built AND again right before the request goes out, and a batch
  that finds the owner active puts everything back for the next pass.
* One flush packs the jobs into a single request (a plain request when only
  one job is due, byte-identical to the old per-job prompt). The reply is a
  JSON object ``{"results": [{"id": n, "output": ...}]}``; each job's
  ``apply`` gets its own output as text. A job with no usable output is
  retried on a later flush; after ``MAX_ATTEMPTS`` it is sent alone, once, as
  it always was, and if that also fails ``apply("")`` runs (the same empty
  answer the per-job path produced on a failure).

An in-flight HTTP request cannot be cancelled, so "abort when the owner
starts talking" means: never START one then, keep a batch small
(``max_jobs`` / ``max_chars``), and log it when the owner spoke during the
call. ``enabled()`` False restores the old behaviour: ``submit`` returns
False and the caller runs its job the way it used to.

Log lines (numbers and tags only, never prompt or speech text):
``[learn-batch] sent N job(s) (a=x b=y) chars=C in T.Ts`` and
``[learn-batch] deferred N job(s): <reason>``.
"""
from __future__ import annotations

import json
import threading
import time

MAX_ATTEMPTS = 3
DEFAULT_INTERVAL_S = 300.0
DEFAULT_MAX_JOBS = 20
DEFAULT_MAX_CHARS = 14000
DEFAULT_POLL_S = 5.0
# Hard ceiling on the queue: past it the OLDEST simple job is answered with ""
# (fail-open) so a day of never-quiet cannot grow it without bound.
QUEUE_HARD_CAP = 200

_WRAP_SYSTEM = (
    "You run several independent small jobs in one reply. Each job names an "
    "instruction set (A, B, ...) and gives its input. Do every job exactly as "
    "its instruction set says, independently of the others, and never mix "
    "their inputs. Reply with ONLY one JSON object, nothing else:\n"
    '{"results": [{"id": <job id>, "output": <that job\'s answer>}, ...]}\n'
    "with exactly one entry per job, in order. Where an instruction set asks "
    "for a JSON object, put that object itself as the output value; where it "
    "asks for a single word, put that word as a JSON string.")


class Job:
    """One queued unit of background learning.

    ``tag``: caller name for the log. ``build()``: returns
    ``(system, user, max_tokens, ctx)`` or None when there is nothing to send
    any more. ``apply(text, ctx)``: gets the answer ("" on failure).
    ``unbuild(ctx)``: gives a lazily built input back (optional)."""

    __slots__ = ("tag", "build", "apply", "unbuild", "queued_at", "attempts",
                 "id")

    def __init__(self, tag, build, apply, unbuild=None):
        self.tag = _clean_tag(tag)
        self.build = build
        self.apply = apply
        self.unbuild = unbuild
        self.queued_at = 0.0
        self.attempts = 0
        self.id = 0

    @classmethod
    def simple(cls, tag, system, user, max_tokens, apply):
        """A job whose prompt is fixed at submit time; ``apply(text)``."""
        spec = (str(system), str(user), int(max_tokens), None)
        return cls(tag, lambda: spec, lambda text, _ctx: apply(text))


def _clean_tag(tag) -> str:
    try:
        t = "_".join(str(tag or "").split())
    except Exception:
        t = ""
    return t[:32] or "job"


def parse_results(text, ids) -> dict:
    """{job id: output text} from a packed reply. Never raises; ids missing
    from the reply (or with a null output) are simply absent."""
    out: dict = {}
    try:
        s = text if isinstance(text, str) else ""
        start = s.find("{")
        if start < 0:
            return out
        data, _ = json.JSONDecoder().raw_decode(s, start)
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return out
        wanted = set(ids)
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                jid = int(row.get("id"))
            except Exception:
                continue
            if jid not in wanted or jid in out:
                continue
            val = row.get("output")
            if val is None:
                continue
            if isinstance(val, str):
                out[jid] = val
            elif isinstance(val, (dict, list)):
                out[jid] = json.dumps(val, ensure_ascii=False)
            else:
                out[jid] = str(val)
    except Exception:
        return out
    return out


def pack(specs) -> "tuple[str, str, int]":
    """(system, user, max_tokens) of one packed request over
    [(job_id, system, user, max_tokens)]. Identical instruction sets are
    written once."""
    sets: dict = {}
    order: list = []
    for _jid, system, _user, _mt in specs:
        if system not in sets:
            sets[system] = chr(ord("A") + len(order)) if len(order) < 26 \
                else f"S{len(order)}"
            order.append(system)
    parts = ["INSTRUCTION SETS"]
    for system in order:
        parts.append(f"=== {sets[system]} ===\n{system}")
    parts.append("JOBS")
    for jid, system, user, _mt in specs:
        parts.append(f"=== JOB {jid} (instruction set {sets[system]}) ===\n"
                     f"{user}")
    budget = sum(int(mt) for *_a, mt in specs) + 20 * len(specs) + 40
    return _WRAP_SYSTEM, "\n\n".join(parts), min(2000, budget)


class LearnBatcher:
    """See the module docstring. All collaborators are injected so the class
    is hermetic: ``llm(system, user, max_tokens) -> str``, ``quiet() -> bool``,
    ``enabled()``, ``interval_s()``, ``max_jobs()``, ``clock()``, ``log(str)``.
    """

    def __init__(self, llm, quiet, *, enabled=None, interval_s=None,
                 max_jobs=None, max_chars=None, clock=None, log=None,
                 poll_s=DEFAULT_POLL_S):
        self._llm = llm
        self._quiet = quiet
        self._enabled = enabled
        self._interval_s = interval_s
        self._max_jobs = max_jobs
        self._max_chars = max_chars
        self._clock = clock or time.monotonic
        self._log = log or print
        self._poll_s = float(poll_s)
        self._lock = threading.Lock()
        self._queue: list = []
        self._seq = 0
        self._flush_lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.stats = {"flushes": 0, "jobs": 0, "deferred": 0, "retried": 0,
                      "solo": 0}

    # -- settings (read at call time, never raise) --------------------------
    @staticmethod
    def _val(src, default):
        try:
            v = src() if callable(src) else src
            return default if v is None else v
        except Exception:
            return default

    def enabled(self) -> bool:
        return bool(self._val(self._enabled, True))

    def interval_s(self) -> float:
        try:
            return max(0.0, float(self._val(self._interval_s,
                                            DEFAULT_INTERVAL_S)))
        except Exception:
            return DEFAULT_INTERVAL_S

    def max_jobs(self) -> int:
        try:
            return max(1, int(self._val(self._max_jobs, DEFAULT_MAX_JOBS)))
        except Exception:
            return DEFAULT_MAX_JOBS

    def max_chars(self) -> int:
        try:
            return max(1000, int(self._val(self._max_chars,
                                           DEFAULT_MAX_CHARS)))
        except Exception:
            return DEFAULT_MAX_CHARS

    def _say(self, msg: str) -> None:
        try:
            self._log(msg)
        except Exception:
            pass

    def _is_quiet(self) -> bool:
        try:
            return bool(self._quiet())
        except Exception:
            return False        # doubt = the owner may be talking: hold off

    # -- queue ----------------------------------------------------------------
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def submit(self, job: Job) -> bool:
        """Queue ``job`` for the next batch. False = batching is off, the
        caller runs the job the old way. Never raises, never blocks."""
        try:
            if not self.enabled():
                return False
            overflow = None
            with self._lock:
                self._seq += 1
                job.id = self._seq
                job.queued_at = float(self._clock())
                self._queue.append(job)
                if len(self._queue) > QUEUE_HARD_CAP:
                    overflow = self._queue.pop(0)
            if overflow is not None:
                self._say("  [learn-batch] queue full - dropping the oldest "
                          f"{overflow.tag} job")
                self._give_up(overflow, None)
            self._ensure_thread()
            return True
        except Exception as e:
            self._say(f"  [learn-batch] submit failed: {type(e).__name__}")
            return False

    def _ensure_thread(self) -> None:
        with self._lock:
            t = self._thread
            if t is not None and t.is_alive():
                return
            if self._stop.is_set():
                return
            self._thread = threading.Thread(
                target=self._loop, name="learn-batch", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._poll_s):
            try:
                self.flush_once()
            except Exception as e:      # the loop must outlive any one pass
                self._say(f"  [learn-batch] flush failed: "
                          f"{type(e).__name__}")

    # -- flushing -------------------------------------------------------------
    def due(self, now=None) -> bool:
        """Is a flush worth attempting? (does not look at quiet)."""
        with self._lock:
            if not self._queue:
                return False
            if len(self._queue) >= self.max_jobs():
                return True
            t = float(self._clock() if now is None else now)
            return (t - self._queue[0].queued_at) >= self.interval_s()

    def flush_once(self, force: bool = False):
        """One flush attempt. Returns None when nothing was due / sent, else
        a summary dict. ``force`` skips the age check (never the quiet one).
        """
        if not self._flush_lock.acquire(blocking=False):
            return None
        try:
            if not force and not self.due():
                return None
            with self._lock:
                if not self._queue:
                    return None
                n_waiting = len(self._queue)
            if not self._is_quiet():
                self.stats["deferred"] += 1
                self._say(f"  [learn-batch] deferred {n_waiting} job(s): "
                          "owner active")
                return None
            return self._send_one_batch()
        finally:
            self._flush_lock.release()

    def _take(self) -> list:
        with self._lock:
            take = self._queue[:self.max_jobs()]
            del self._queue[:len(take)]
            return take

    def _requeue(self, jobs) -> None:
        with self._lock:
            self._queue[0:0] = jobs

    def _send_one_batch(self):
        started = self._clock()
        taken = self._take()
        built = []          # (job, system, user, max_tokens, ctx)
        for job in taken:
            try:
                spec = job.build()
            except Exception as e:
                self._say(f"  [learn-batch] {job.tag} build failed: "
                          f"{type(e).__name__}")
                spec = None
            if spec is None:
                continue
            built.append((job,) + tuple(spec))
        if not built:
            return None
        # Greedy size cap: the rest goes back (lazy jobs hand their input
        # back) and rides the next pass.
        kept, size = [], 0
        for item in built:
            add = len(item[1]) + len(item[2])
            if kept and size + add > self.max_chars():
                break
            kept.append(item)
            size += add
        spill = built[len(kept):]
        if spill:
            self._unbuild_all(spill)
            self._requeue([it[0] for it in spill])
        built = kept
        # Owner turns always win: re-check right before the request goes out.
        if not self._is_quiet():
            self._unbuild_all(built)
            self._requeue([it[0] for it in built])
            self.stats["deferred"] += 1
            self._say(f"  [learn-batch] deferred {len(built)} job(s): owner "
                      "became active before the send")
            return None
        counts: dict = {}
        for it in built:
            counts[it[0].tag] = counts.get(it[0].tag, 0) + 1
        mix = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if len(built) == 1:
            job, system, user, max_tokens, _ctx = built[0]
            call = (system, user, int(max_tokens))
            packed = False
        else:
            call = pack([(it[0].id, it[1], it[2], it[3]) for it in built])
            packed = True
        chars = len(call[0]) + len(call[1])
        text = ""
        try:
            text = self._llm(*call) or ""
        except Exception as e:
            self._say(f"  [learn-batch] call failed: {type(e).__name__}")
            text = ""
        owner_during = not self._is_quiet()
        results = ({built[0][0].id: text} if (not packed and text.strip())
                   else parse_results(text, [it[0].id for it in built]))
        ok = fail = 0
        retry: list = []
        for job, _s, _u, _mt, ctx in built:
            out = results.get(job.id)
            if out is not None and (out.strip() or not packed):
                self._deliver(job, out, ctx)
                ok += 1
                continue
            fail += 1
            job.attempts += 1
            if job.attempts >= MAX_ATTEMPTS:
                self._solo(job, _s, _u, _mt, ctx)
            else:
                self._unbuild_all([(job, _s, _u, _mt, ctx)])
                job.queued_at = float(self._clock())    # back off one interval
                retry.append(job)
        if retry:
            self.stats["retried"] += len(retry)
            self._requeue(retry)
        elapsed = float(self._clock()) - float(started)
        self.stats["flushes"] += 1
        self.stats["jobs"] += ok
        extra = (f" retry={len(retry)}" if retry else "") \
            + (" owner-spoke-during-call" if owner_during else "")
        self._say(f"  [learn-batch] sent {len(built)} job(s) ({mix}) "
                  f"packed={'yes' if packed else 'no'} ok={ok} fail={fail} "
                  f"chars={chars} in {elapsed:.1f}s{extra}")
        return {"jobs": len(built), "ok": ok, "failed": fail,
                "retried": len(retry), "chars": chars, "elapsed_s": elapsed,
                "packed": packed, "owner_during": owner_during}

    # -- helpers --------------------------------------------------------------
    def _unbuild_all(self, items) -> None:
        for job, _s, _u, _mt, ctx in items:
            try:
                if job.unbuild is not None:
                    job.unbuild(ctx)
            except Exception as e:
                self._say(f"  [learn-batch] {job.tag} unbuild failed: "
                          f"{type(e).__name__}")

    def _deliver(self, job, text, ctx) -> None:
        try:
            job.apply(text, ctx)
        except Exception as e:
            self._say(f"  [learn-batch] {job.tag} apply failed: "
                      f"{type(e).__name__}")

    def _solo(self, job, system, user, max_tokens, ctx) -> None:
        """Last resort for a job the packed call keeps failing: the old
        one-request-per-job path, once, only while the owner is quiet. An
        empty answer is what a failed single call always returned."""
        self.stats["solo"] += 1
        text = ""
        if self._is_quiet():
            try:
                text = self._llm(system, user, int(max_tokens)) or ""
            except Exception:
                text = ""
        self._deliver(job, text, ctx)

    def _give_up(self, job, ctx) -> None:
        self._deliver(job, "", ctx)
