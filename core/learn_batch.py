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
  retried on a later flush (one interval later, never in the same pass);
  after ``MAX_ATTEMPTS`` it is sent alone, once, as it always was, and if
  that also fails ``apply("")`` runs (the same empty answer the per-job path
  produced on a failure).

FAILED PACKED BATCHES (v3.0.26, overhaul finding reliability-07)
================================================================
The 11-16/17 logs had 14 packed batches with ok=0 - every job of the reply
unreadable - and the retry re-packed the IDENTICAL batch, which failed the
same way (16:15:44 and 16:20:54: the same 10 jobs, ok=0 both times). The
replies were not short: each failed call generated 14-20 tokens per job,
the same as a call that worked, so the model answered every job but in a
shape the parser could not read. What the failures had in common was size:
packed prompts with a learn job failed 1 time in 58 under 9,500 characters,
8 in 43 from 9,500 to 10,500 and 5 in 5 above. The format instruction sits
at the very top of the system prompt, ~10,000 characters before the point
where the model starts writing. So:

* ``pack()`` restates the reply format, with the exact job ids, at the END
  of the request, next to where the answer starts;
* a packed batch whose non-empty reply yields NO job is split: each of its
  jobs carries a character cap of half that request (``Job.cap``) into the
  next pass, so the retry is a smaller request, never the same one; the job
  that keeps failing ends up alone, and alone it is dropped (``apply("")``)
  after ``MAX_ATTEMPTS`` instead of being sent a fourth time;
* the first such reply in an hour is logged by its SHAPE only (words masked,
  ``_reply_shape``) so the next session shows what the model wrote without a
  word of the owner's speech in the log.

An in-flight HTTP request cannot be cancelled, so "abort when the owner
starts talking" means: never START one then, keep a batch small
(``max_jobs`` / ``max_chars``), and log it when the owner spoke during the
call. v3.0.27 (core/bg_delivery.py): "spoke during the call" also counts an
owner turn accepted while the call ran (``owner_mark()`` moved), and a job
marked ``spoken`` (its answer ends in a line said aloud) is then answered ""
instead of with its output when ``skip_spoken()`` (BG_SKIP_IF_OWNER_SPOKE)
is on; data jobs (learning records) are always delivered. ``enabled()`` False restores the old behaviour: ``submit`` returns
False and the caller runs its job the way it used to.

Log lines (numbers and tags only, never prompt or speech text):
``[learn-batch] sent N job(s) (a=x b=y) chars=C in T.Ts`` (with
``split<=N`` when a failed packed request is split, ``dropped=N`` when a
job is answered "" after its last attempt),
``[learn-batch] deferred N job(s): <reason>`` (for "owner active": once per
streak of deferred polls, then ``owner quiet again after N deferred passes``),
and, at most hourly,
``[learn-batch] unreadable packed reply (N job(s), C chars) shape: ...``
(words masked by _reply_shape).
"""
from __future__ import annotations

import json
import re
import threading
import time

MAX_ATTEMPTS = 3
DEFAULT_INTERVAL_S = 300.0
DEFAULT_MAX_JOBS = 20
DEFAULT_MAX_CHARS = 14000
DEFAULT_POLL_S = 5.0
# A split never asks for a request smaller than this (one learn job alone is
# ~8,000 characters, and the greedy cap always keeps the first job anyway).
SPLIT_MIN_CHARS = 1000
# Job.cap of a job that must go out alone on its next pass.
SOLO_CAP = 1
# At most one "unreadable packed reply" shape line per this many seconds.
SHAPE_LOG_EVERY_S = 3600.0
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
    ``unbuild(ctx)``: gives a lazily built input back (optional).
    ``cap``: 0, or the most characters the next request carrying this job
    may hold (set when a packed request with it failed outright; SOLO_CAP =
    it goes out alone). ``spoken``: its answer ends in a line said aloud, so
    it is answered "" when the owner spoke during the call (v3.0.27)."""

    __slots__ = ("tag", "build", "apply", "unbuild", "queued_at", "attempts",
                 "id", "cap", "spoken")

    def __init__(self, tag, build, apply, unbuild=None, spoken=False):
        self.tag = _clean_tag(tag)
        self.build = build
        self.apply = apply
        self.unbuild = unbuild
        self.queued_at = 0.0
        self.attempts = 0
        self.id = 0
        self.cap = 0
        self.spoken = bool(spoken)

    @classmethod
    def simple(cls, tag, system, user, max_tokens, apply, spoken=False):
        """A job whose prompt is fixed at submit time; ``apply(text)``."""
        spec = (str(system), str(user), int(max_tokens), None)
        return cls(tag, lambda: spec, lambda text, _ctx: apply(text),
                   spoken=spoken)


def _clean_tag(tag) -> str:
    try:
        t = "_".join(str(tag or "").split())
    except Exception:
        t = ""
    return t[:32] or "job"


def _job_id(value):
    """A job id the model wrote: 3, "3", "JOB 3", "job_3", "Job #3" -> 3;
    anything else -> None. (The small Unraid model writes "JOB 3"; the old
    int() rejected it and every job of the batch counted as failed.)"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    m = _JOB_ID_RE.match(str(value or "").strip())
    return int(m.group(1)) if m else None


def _output_text(val):
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _rows_one_by_one(s: str) -> list:
    """(id, output) for every '"id": N, "output": <value>' in ``s``, each
    value decoded on its own. Rescues a reply whose OUTER JSON is broken:
    gemma4:26b regularly drops one closing brace at the very end of a packed
    reply, which made the whole reply unparseable and failed 452 of 708 packed
    jobs in the 2026-10-16..26 live logs (each then retried, then sent alone:
    2-3x the background calls, each evicting the voice brain's prompt)."""
    rows = []
    dec = json.JSONDecoder()
    pos = 0
    while True:
        m = _ROW_RE.search(s, pos)
        if m is None:
            return rows
        try:
            val, pos = dec.raw_decode(s, m.end())
        except ValueError:
            pos = m.end()
            continue
        # The scan resumes AFTER the decoded value, so a row written inside
        # another job's output is never read as a row (security review LB-1).
        rows.append((m.group(1), val))


_JOB_ID_RE = re.compile(r"(?:job\s*[#_-]?\s*)?(\d+)\Z", re.IGNORECASE)
_ROW_RE = re.compile(
    r'"id"\s*:\s*("[^"\n]{1,20}"|\d+)\s*,\s*"output"\s*:\s*')


def parse_results(text, ids) -> dict:
    """{job id: output text} from a packed reply. Never raises; ids missing
    from the reply (or with a null output) are simply absent.

    The whole reply is decoded first. When that fails or misses ids, each
    '"id": N, "output": ...' entry is decoded on its own (see
    _rows_one_by_one), so one malformed bracket no longer loses every job.
    Ids are accepted as 3, "3" or "JOB 3"."""
    out: dict = {}
    wanted = set(ids)

    def take(raw_id, val) -> None:
        jid = _job_id(raw_id.strip('"') if isinstance(raw_id, str)
                      and raw_id.startswith('"') else raw_id)
        if jid is None or jid not in wanted or jid in out:
            return
        txt = _output_text(val)
        if txt is not None:
            out[jid] = txt

    s = text if isinstance(text, str) else ""
    try:
        start = s.find("{")
        if start < 0:
            return out
        try:
            data, _ = json.JSONDecoder().raw_decode(s, start)
        except ValueError:
            data = None
        rows = data.get("results") if isinstance(data, dict) else None
        present = set()
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    present.add(_job_id(row.get("id")))
                    take(row.get("id"), row.get("output"))
        # The rescue only fills ids the whole decode never saw (an id present
        # with a null output stays absent), and an id with more than one
        # candidate row is left out to retry alone: never pick the first of
        # several (security review LB-1).
        if wanted - set(out) - present:
            cands = [(_job_id(r.strip('"') if isinstance(r, str) else r), v)
                     for r, v in _rows_one_by_one(s)]
            counts: dict = {}
            for jid, _v in cands:
                counts[jid] = counts.get(jid, 0) + 1
            for jid, val in cands:
                if jid not in present and counts.get(jid) == 1:
                    take(jid, val)
    except Exception:
        return out
    return out


def _format_reminder(ids) -> str:
    """The reply format again, at the END of a packed request (see the
    module docstring: on a ~10,000-character request the system prompt's
    format line is too far from where the model starts writing)."""
    listed = ", ".join(str(i) for i in ids)
    return ("REPLY FORMAT (all jobs above)\n"
            'Now reply with ONLY one JSON object, {"results": [...]}, holding '
            f'exactly one {{"id": <job id>, "output": <answer>}} entry for each '
            f"of the {len(ids)} jobs, in this order: ids {listed}. "
            "No other text before or after it.")


def pack(specs) -> "tuple[str, str, int]":
    """(system, user, max_tokens) of one packed request over
    [(job_id, system, user, max_tokens)]. Identical instruction sets are
    written once; the reply format is restated last (_format_reminder)."""
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
    parts.append(_format_reminder([jid for jid, *_r in specs]))
    budget = sum(int(mt) for *_a, mt in specs) + 20 * len(specs) + 40
    return _WRAP_SYSTEM, "\n\n".join(parts), min(2000, budget)


_SHAPE_KEEP = frozenset({"results", "id", "output", "job", "null", "true",
                         "false"})
_SHAPE_WORD_RE = re.compile(r"[^\W\d_]+")
_SHAPE_DIGITS_RE = re.compile(r"\d+")


def _reply_shape(text, limit: int = 160) -> str:
    """A reply's SHAPE for the log: every word except the format's own
    (results / id / output / job / null / true / false) becomes "w", every
    run of more than three digits "9", whitespace collapses, at most
    ``limit`` characters. Shows how the model laid the reply out without a
    word of what it said (the outputs are extracted from the owner's
    speech). Never raises."""
    try:
        s = " ".join(str(text or "").split())

        def word(m):
            return m.group(0) if m.group(0).lower() in _SHAPE_KEEP else "w"

        def digits(m):
            return m.group(0) if len(m.group(0)) <= 3 else "9"
        s = _SHAPE_DIGITS_RE.sub(digits, _SHAPE_WORD_RE.sub(word, s))
        return s[:max(0, int(limit))]
    except Exception:
        return ""


class LearnBatcher:
    """See the module docstring. All collaborators are injected so the class
    is hermetic: ``llm(system, user, max_tokens) -> str``, ``quiet() -> bool``,
    ``enabled()``, ``interval_s()``, ``max_jobs()``, ``clock()``, ``log(str)``,
    ``owner_mark() -> int`` (accepted owner turns so far; None = not
    tracked) and ``skip_spoken() -> bool`` (BG_SKIP_IF_OWNER_SPOKE; None =
    off, the v3.0.26 behaviour).
    """

    def __init__(self, llm, quiet, *, enabled=None, interval_s=None,
                 max_jobs=None, max_chars=None, clock=None, log=None,
                 poll_s=DEFAULT_POLL_S, owner_mark=None, skip_spoken=None):
        self._llm = llm
        self._quiet = quiet
        self._enabled = enabled
        self._interval_s = interval_s
        self._max_jobs = max_jobs
        self._max_chars = max_chars
        self._clock = clock or time.monotonic
        self._log = log or print
        self._poll_s = float(poll_s)
        self._owner_mark = owner_mark
        self._skip_spoken = skip_spoken
        self._lock = threading.Lock()
        self._queue: list = []
        self._seq = 0
        self._flush_lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.stats = {"flushes": 0, "jobs": 0, "deferred": 0, "retried": 0,
                      "solo": 0, "split": 0, "dropped": 0, "skipped": 0}
        self._shape_logged_at = None     # clock() of the last shape line
        # "owner active" deferrals in a row (v3.0.26): logged once at the
        # start of a streak and once, with the count, when it ends - the
        # 5 s poll used to print a line every pass (183 in one hour).
        self._defer_streak = 0

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

    def _mark(self):
        """owner_mark(), or None when not tracked / unreadable."""
        if self._owner_mark is None:
            return None
        try:
            return int(self._owner_mark())
        except Exception:
            return None

    def _skip_spoken_on(self) -> bool:
        if self._skip_spoken is None:
            return False
        try:
            return bool(self._skip_spoken())
        except Exception:
            return False

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
                self._defer_streak += 1
                if self._defer_streak == 1:
                    self._say(f"  [learn-batch] deferred {n_waiting} job(s): "
                              "owner active")
                return None
            if self._defer_streak > 1:
                self._say(f"  [learn-batch] owner quiet again after "
                          f"{self._defer_streak} deferred passes")
            self._defer_streak = 0
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
        # back) and rides the next pass. A job split out of a failed packed
        # request (Job.cap) lowers the cap of any request it is in, so the
        # retry is smaller than the request that failed; SOLO_CAP = alone.
        kept, size = [], 0
        limit = self.max_chars()
        for item in built:
            add = len(item[1]) + len(item[2])
            cap = int(getattr(item[0], "cap", 0) or 0)
            item_limit = min(limit, cap) if cap > 0 else limit
            if kept and size + add > item_limit:
                break
            kept.append(item)
            size += add
            limit = item_limit
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
        mark = self._mark()
        try:
            text = self._llm(*call) or ""
        except Exception as e:
            self._say(f"  [learn-batch] call failed: {type(e).__name__}")
            text = ""
        owner_during = not self._is_quiet()
        if not owner_during and mark is not None:
            owner_during = self._mark() != mark
        skip_spoken = owner_during and self._skip_spoken_on()
        results = ({built[0][0].id: text} if (not packed and text.strip())
                   else parse_results(text, [it[0].id for it in built]))
        if packed and text.strip() and not any(
                results.get(it[0].id, "").strip() for it in built):
            # The model answered but not one job could be read: re-packing
            # the same request fails the same way (reliability-07), so split.
            split_cap = max(SPLIT_MIN_CHARS, chars // 2)
            self._note_unreadable(text, len(built))
        else:
            split_cap = 0
        ok = fail = dropped = skipped = 0
        retry: list = []
        for job, _s, _u, _mt, ctx in built:
            out = results.get(job.id)
            if out is not None and (out.strip() or not packed):
                if skip_spoken and getattr(job, "spoken", False):
                    # v3.0.27: its line would land over / right after his
                    # turn. Answered "" (the failure answer), never retried.
                    skipped += 1
                    self._give_up(job, ctx)
                    continue
                self._deliver(job, out, ctx)
                ok += 1
                continue
            fail += 1
            job.attempts += 1
            if job.attempts >= MAX_ATTEMPTS and not packed:
                # It already went out ALONE and failed at its last attempt:
                # the old per-job failure answer, not a fourth identical call.
                dropped += 1
                self._give_up(job, ctx)
                continue
            self._unbuild_all([(job, _s, _u, _mt, ctx)])
            job.queued_at = float(self._clock())    # back off one interval
            if job.attempts >= MAX_ATTEMPTS:
                job.cap = SOLO_CAP                  # alone, on the next pass
            elif split_cap:
                job.cap = min(job.cap, split_cap) if job.cap else split_cap
            retry.append(job)
        if retry:
            self.stats["retried"] += len(retry)
            self._requeue(retry)
        if split_cap:
            self.stats["split"] += 1
        self.stats["dropped"] += dropped
        self.stats["skipped"] += skipped
        elapsed = float(self._clock()) - float(started)
        self.stats["flushes"] += 1
        self.stats["jobs"] += ok
        if not packed and getattr(built[0][0], "cap", 0) == SOLO_CAP:
            self.stats["solo"] += 1
        extra = (f" retry={len(retry)}" if retry else "") \
            + (f" split<={split_cap}" if split_cap else "") \
            + (f" dropped={dropped}" if dropped else "") \
            + (f" skipped-spoken={skipped}" if skipped else "") \
            + (" owner-spoke-during-call" if owner_during else "")
        self._say(f"  [learn-batch] sent {len(built)} job(s) ({mix}) "
                  f"packed={'yes' if packed else 'no'} ok={ok} fail={fail} "
                  f"chars={chars} in {elapsed:.1f}s{extra}")
        return {"jobs": len(built), "ok": ok, "failed": fail,
                "retried": len(retry), "chars": chars, "elapsed_s": elapsed,
                "packed": packed, "owner_during": owner_during,
                "split_cap": split_cap, "dropped": dropped,
                "skipped": skipped}

    def _note_unreadable(self, text, n_jobs: int) -> None:
        """At most once per SHAPE_LOG_EVERY_S: the shape of a packed reply
        none of whose jobs could be read (_reply_shape: words masked)."""
        try:
            now = float(self._clock())
            last = self._shape_logged_at
            if last is not None and (now - last) < SHAPE_LOG_EVERY_S:
                return
            self._shape_logged_at = now
            self._say(f"  [learn-batch] unreadable packed reply ({n_jobs} "
                      f"job(s), {len(text)} chars) shape: "
                      f"{_reply_shape(text)}")
        except Exception:
            pass

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

    def _give_up(self, job, ctx) -> None:
        self._deliver(job, "", ctx)
