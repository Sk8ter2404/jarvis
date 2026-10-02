"""core/local_traffic.py -- background gate, thread tag, own-inference tracker.

WHY (2026-09-29, live)
======================
Ollama serves the local brain from ONE slot. Every background local request
between two owner turns (learn_from_turn's extraction, the ambient extractor,
a Teams-nudger screenshot read by the vision model) replaced the cached
~12.5k-token prefix, so every owner turn paid a ~2.2 s full prompt
re-evaluation. The fix defers NON-URGENT background requests while the owner
is in a conversation. These tests pin the contracts the monolith relies on:

* a tagged job waits while the predicate says busy and runs once it clears;
* jobs run ONE AT A TIME in arrival order (a burst never queues in Ollama in
  front of the owner's next turn);
* every wait is bounded -- the soft cap forces a run between turns, the hard
  grace forces it regardless -- and a cancel ends the wait at once;
* an untagged thread (the owner) and the main thread NEVER wait;
* the slot is re-entrant per thread (a job's nested local calls can't
  deadlock on themselves);
* a broken predicate never holds a job back;
* the tracker reports in-flight / recently-finished inference and counts
  POSTs, excluding the re-prime's own.

Stdlib only: runs in the CI-light tier. Every class fails on the pre-r6 tree
(the module does not exist).
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from core import local_traffic as lt


class _Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def _gate(reason_box, cap=5.0, grace=2.0, poll=0.01, log=None):
    return lt.BackgroundGate(lambda: reason_box[0], lambda: cap,
                             hard_grace_s=grace, poll_s=poll,
                             log=log if log is not None else (lambda m: None))


def _run_in_thread(fn, *args, **kwargs):
    out = {}

    def _t():
        try:
            out["value"] = fn(*args, **kwargs)
        except BaseException as e:  # pragma: no cover - surfaced below
            out["error"] = e
    th = threading.Thread(target=_t, daemon=True)
    th.start()
    return th, out


class BackgroundWorkTagTests(unittest.TestCase):
    def test_untagged_by_default_and_tag_is_scoped(self):
        self.assertIsNone(lt.current_job())
        with lt.background_work("learn from turn") as job:
            self.assertIs(lt.current_job(), job)
            self.assertEqual(job.tag, "learn_from_turn",
                             "a tag must stay one whitespace-free token")
        self.assertIsNone(lt.current_job())

    def test_nested_keeps_the_outer_job_and_its_start(self):
        clock = _Clock(10.0)
        with lt.background_work("outer", clock=clock) as outer:
            clock.t = 50.0
            with lt.background_work("inner", clock=clock) as inner:
                self.assertIs(inner, outer)
                self.assertEqual(lt.current_job().started_at, 10.0)
            self.assertIs(lt.current_job(), outer)
        self.assertIsNone(lt.current_job())

    def test_tag_is_per_thread(self):
        seen = {}
        with lt.background_work("mine"):
            th, _ = _run_in_thread(lambda: seen.update(job=lt.current_job()))
            th.join(5)
        self.assertIsNone(seen["job"])

    def test_tag_is_cleared_when_the_body_raises(self):
        with self.assertRaises(RuntimeError):
            with lt.background_work("x"):
                raise RuntimeError("boom")
        self.assertIsNone(lt.current_job())


class GateBasicsTests(unittest.TestCase):
    def test_quiet_gate_goes_at_once_and_holds_until_release(self):
        box = [None]
        g = _gate(box)
        held, done, out = threading.Event(), threading.Event(), {}

        def holder():
            out["p"] = g.acquire("job")
            held.set()
            done.wait(5)                # a LIVE holder keeps the slot
            g.release(out["p"])

        th = threading.Thread(target=holder, daemon=True)
        th.start()
        self.assertTrue(held.wait(5))
        p = out["p"]
        self.assertEqual(p.outcome, "go")
        self.assertTrue(p.holds)
        self.assertTrue(g.busy())
        # release() must come from the holder's thread
        th2, _ = _run_in_thread(g.release, p)
        th2.join(5)
        self.assertTrue(g.busy(), "a foreign thread released the slot")
        done.set()
        th.join(5)
        self.assertFalse(g.busy(), "the holder's own release must free it")

    def test_reused_thread_ident_cannot_release_the_slot(self):
        # CI 2026-09-29: the gate keyed its holder on threading.get_ident(),
        # which Linux hands to the next thread as soon as one exits, so the
        # "foreign thread" in the test above sometimes WAS the holder by ident.
        # Force the reuse deterministically: every thread gets the same ident.
        box = [None]
        g = _gate(box)
        with mock.patch.object(lt.threading, "get_ident", lambda: 4242):
            hold = threading.Event()
            done = threading.Event()
            out = {}

            def holder():
                out["p"] = g.acquire("job")
                hold.set()
                done.wait(5)            # keep holding (thread stays alive)

            th = threading.Thread(target=holder, daemon=True)
            th.start()
            self.assertTrue(hold.wait(5))
            th2, _ = _run_in_thread(g.release, out["p"])
            th2.join(5)
            self.assertTrue(g.busy(), "a thread with a reused ident released the slot")
            done.set()
            th.join(5)

    def test_a_holder_that_died_without_releasing_frees_the_slot(self):
        box = [None]
        g = _gate(box, cap=30.0, grace=30.0)
        th, out = _run_in_thread(g.acquire, "job")     # acquires, never releases
        th.join(5)
        self.assertTrue(out["value"].holds)
        self.assertFalse(g.busy(), "a dead holder kept the slot")
        t0 = time.monotonic()
        th2, out2 = _run_in_thread(g.acquire, "next")
        th2.join(5)
        self.assertEqual(out2["value"].outcome, "go")
        self.assertLess(time.monotonic() - t0, 2.0)    # no waiting out the 30 s cap

    def test_main_thread_never_waits(self):
        box = ["turn"]
        g = _gate(box, cap=60.0, grace=60.0)
        t0 = time.monotonic()
        p = g.acquire("main")                 # the test runs on the main thread
        self.assertEqual(p.outcome, "off")
        self.assertFalse(p.holds)
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_zero_cap_is_off(self):
        box = ["conversation"]
        g = lt.BackgroundGate(lambda: box[0], lambda: 0.0, poll_s=0.01,
                              log=lambda m: None)
        th, out = _run_in_thread(g.acquire, "job")
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(out["value"].outcome, "off")
        self.assertFalse(g.busy())

    def test_broken_predicate_never_holds_a_job_back(self):
        def _boom():
            raise RuntimeError("predicate bug")
        g = lt.BackgroundGate(_boom, lambda: 60.0, poll_s=0.01,
                              log=lambda m: None)
        th, out = _run_in_thread(g.acquire, "job")
        th.join(5)
        self.assertEqual(out["value"].outcome, "go")


class GateDeferralTests(unittest.TestCase):
    def test_waits_while_busy_then_runs_when_quiet(self):
        box = ["conversation"]
        logs = []
        g = _gate(box, cap=30.0, log=logs.append)
        th, out = _run_in_thread(g.acquire, "learn_from_turn")
        time.sleep(0.15)
        self.assertTrue(th.is_alive(), "ran while the owner was talking")
        box[0] = None
        th.join(5)
        p = out["value"]
        self.assertEqual(p.outcome, "released")
        self.assertTrue(any("[bg-local] defer learn_from_turn (conversation)"
                            in m for m in logs), logs)
        self.assertTrue(any(m.startswith("  [bg-local] run learn_from_turn "
                                         "after ") for m in logs), logs)

    def test_soft_cap_forces_a_run_between_turns(self):
        box = ["conversation"]
        g = _gate(box, cap=0.2, grace=30.0)
        t0 = time.monotonic()
        th, out = _run_in_thread(g.acquire, "job")
        th.join(5)
        self.assertEqual(out["value"].outcome, "forced")
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_past_the_soft_cap_it_still_waits_out_a_hard_reason(self):
        box = ["turn"]
        g = _gate(box, cap=0.1, grace=30.0)
        th, out = _run_in_thread(g.acquire, "job")
        time.sleep(0.4)
        self.assertTrue(th.is_alive(),
                        "forced into the middle of the owner's turn")
        box[0] = "conversation"          # the turn ended: now soft
        th.join(5)
        self.assertEqual(out["value"].outcome, "forced")

    def test_hard_grace_bounds_even_a_hard_reason(self):
        box = ["utterance"]
        g = _gate(box, cap=0.1, grace=0.2)
        t0 = time.monotonic()
        th, out = _run_in_thread(g.acquire, "job")
        th.join(5)
        self.assertFalse(th.is_alive(), "unbounded wait")
        self.assertEqual(out["value"].outcome, "forced")
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_cancel_ends_the_wait(self):
        box = ["conversation"]
        stop = threading.Event()
        g = _gate(box, cap=60.0, grace=60.0)
        th, out = _run_in_thread(g.acquire, "job", cancel=stop.is_set)
        time.sleep(0.1)
        stop.set()
        th.join(5)
        self.assertEqual(out["value"].outcome, "cancelled")

    def test_started_at_counts_toward_the_cap(self):
        # A job that already waited (e.g. before its capture) is not granted
        # a fresh full deferral at its POST.
        box = ["conversation"]
        g = _gate(box, cap=5.0)
        th, out = _run_in_thread(g.acquire, "job",
                                 started_at=time.monotonic() - 10.0)
        th.join(5)
        self.assertEqual(out["value"].outcome, "forced")


class GateOrderTests(unittest.TestCase):
    def test_jobs_run_one_at_a_time_in_arrival_order(self):
        box = ["conversation"]
        g = _gate(box, cap=30.0)
        order, active, peak = [], [0], [0]
        lock = threading.Lock()

        def _job(name):
            p = g.acquire(name)
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                order.append(name)
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            g.release(p)

        threads = []
        for name in ("a", "b", "c"):
            th = threading.Thread(target=_job, args=(name,), daemon=True)
            th.start()
            threads.append(th)
            time.sleep(0.05)             # a strict arrival order
        self.assertEqual(g.waiting(), 3)
        box[0] = None
        for th in threads:
            th.join(5)
        self.assertEqual(order, ["a", "b", "c"])
        self.assertEqual(peak[0], 1, "two background POSTs ran at once")
        self.assertFalse(g.busy())

    def test_reentrant_on_the_holding_thread(self):
        box = [None]
        g = _gate(box)

        def _nested():
            p1 = g.acquire("outer")
            p2 = g.acquire("inner")      # e.g. _llm_quick -> _call_local_llm
            r = (p1.outcome, p2.outcome, g.busy())
            g.release(p2)
            still = g.busy()
            g.release(p1)
            return r + (still, g.busy())
        th, out = _run_in_thread(_nested)
        th.join(5)
        self.assertFalse(th.is_alive(), "deadlocked on its own slot")
        self.assertEqual(out["value"], ("go", "reentrant", True, True, False))


class SlotHelperTests(unittest.TestCase):
    def test_untagged_thread_passes_straight_through(self):
        box = ["turn"]
        g = _gate(box, cap=60.0, grace=60.0)

        def _owner_call():
            with lt.slot(g) as p:
                return p
        th, out = _run_in_thread(_owner_call)
        th.join(5)
        self.assertFalse(th.is_alive(), "an owner call waited")
        self.assertIsNone(out["value"])

    def test_tagged_thread_waits_and_releases(self):
        box = ["conversation"]
        g = _gate(box, cap=30.0)

        def _bg():
            with lt.background_work("teams-nudge"):
                with lt.slot(g) as p:
                    held = g.busy()
                return p.outcome, held, g.busy()
        th, out = _run_in_thread(_bg)
        time.sleep(0.1)
        self.assertTrue(th.is_alive())
        box[0] = None
        th.join(5)
        self.assertEqual(out["value"], ("released", True, False))

    def test_wait_for_quiet_does_not_keep_the_slot(self):
        box = [None]
        g = _gate(box)

        def _bg():
            with lt.background_work("teams-nudge"):
                outcome = lt.wait_for_quiet(g)
            return outcome, g.busy()
        th, out = _run_in_thread(_bg)
        th.join(5)
        self.assertEqual(out["value"], ("go", False))
        self.assertEqual(lt.wait_for_quiet(g), "none")   # main thread

    def test_default_gate_never_waits_without_a_predicate(self):
        g = lt.BackgroundGate(log=lambda m: None)

        def _bg():
            with lt.background_work("x"):
                with lt.slot(g) as p:
                    return p.outcome
        th, out = _run_in_thread(_bg)
        th.join(5)
        self.assertEqual(out["value"], "go")


class TrackerTests(unittest.TestCase):
    def test_in_flight_then_recent_then_idle(self):
        clock = _Clock(100.0)
        tr = lt.InferenceTracker(clock=clock)
        self.assertFalse(tr.busy_within(10.0))
        with tr.track():
            self.assertEqual(tr.inflight, 1)
            clock.t = 500.0
            self.assertTrue(tr.busy_within(10.0), "in flight must count")
        self.assertEqual(tr.inflight, 0)
        clock.t = 509.0
        self.assertTrue(tr.busy_within(10.0))
        clock.t = 511.0
        self.assertFalse(tr.busy_within(10.0))

    def test_posts_count_excludes_uncounted(self):
        tr = lt.InferenceTracker()
        with tr.track():
            pass
        with tr.track(count=False):       # the idle re-prime
            pass
        self.assertEqual(tr.posts, 1)

    def test_track_survives_a_raising_post(self):
        tr = lt.InferenceTracker()
        with self.assertRaises(ValueError):
            with tr.track():
                raise ValueError("timeout")
        self.assertEqual(tr.inflight, 0)
        self.assertTrue(tr.busy_within(10.0))

    def test_module_helper_reads_the_shared_tracker(self):
        lt.TRACKER.reset()
        self.assertFalse(lt.own_inference_recent(10.0))
        with lt.TRACKER.track():
            self.assertTrue(lt.own_inference_recent(10.0))
        lt.TRACKER.reset()


# ════════════════════════════════════════════════════════════════════════════
#  Speed plan R5 (2026-10-02): opt-in tags, the strict switch, per-job caps
# ════════════════════════════════════════════════════════════════════════════
def _strict_gate(reason_box, strict, cap=30.0, grace=2.0, log=None):
    return lt.BackgroundGate(lambda: reason_box[0], lambda: cap,
                             hard_grace_s=grace, poll_s=0.01,
                             log=log if log is not None else (lambda m: None),
                             strict=strict)


def _bg_slot(g, tag, **kw):
    """On this (non-main) thread: run one tagged job's slot; returns
    (outcome, slot held inside, slot held after)."""
    with lt.background_work(tag, **kw):
        with lt.slot(g) as p:
            held = g.busy()
        return (p.outcome if p is not None else None), held, g.busy()


class OptInShadowTests(unittest.TestCase):
    """An opt-in job with the strict switch OFF behaves exactly as untagged
    (no wait, no slot) and only LOGS, once, that it would have waited."""

    def test_switch_off_runs_at_once_mid_turn_and_logs_once(self):
        box, logs = ["turn"], []
        g = _strict_gate(box, strict=lambda: False, log=logs.append)

        def _job():
            with lt.background_work("chappie", opt_in=True):
                with lt.slot(g) as p1:
                    first = (p1.outcome, p1.holds, g.busy())
                with lt.slot(g) as p2:       # a second POST of the same job
                    second = p2.outcome
            return first, second
        t0 = time.monotonic()
        th, out = _run_in_thread(_job)
        th.join(5)
        self.assertFalse(th.is_alive(), "a shadow job waited")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(out["value"], (("shadow", False, False), "shadow"))
        self.assertEqual(logs, ["  [bg-local] shadow chappie would defer "
                                "(turn)"],
                         "one line per job, numbers and tags only")

    def test_switch_off_and_quiet_logs_nothing(self):
        box, logs = [None], []
        g = _strict_gate(box, strict=lambda: False, log=logs.append)
        th, out = _run_in_thread(_bg_slot, g, "chappie", opt_in=True)
        th.join(5)
        self.assertEqual(out["value"], ("shadow", False, False))
        self.assertEqual(logs, [])

    def test_disabled_gate_logs_no_shadow(self):
        box, logs = ["turn"], []
        g = _strict_gate(box, strict=lambda: False, cap=0.0, log=logs.append)
        th, _ = _run_in_thread(_bg_slot, g, "chappie", opt_in=True)
        th.join(5)
        self.assertEqual(logs, [], "a disabled gate would hold nothing")

    def test_switch_on_waits_like_any_tagged_job(self):
        box = ["turn"]
        g = _strict_gate(box, strict=lambda: True)
        th, out = _run_in_thread(_bg_slot, g, "notification-triage",
                                 opt_in=True)
        time.sleep(0.15)
        self.assertTrue(th.is_alive(), "ran in the middle of the owner's turn")
        box[0] = None
        th.join(5)
        self.assertEqual(out["value"], ("released", True, False))

    def test_unconfigured_switch_means_wait(self):
        box = ["utterance"]
        g = _strict_gate(box, strict=None)
        self.assertTrue(g.strict())
        th, out = _run_in_thread(_bg_slot, g, "chappie", opt_in=True)
        time.sleep(0.15)
        self.assertTrue(th.is_alive())
        box[0] = None
        th.join(5)
        self.assertEqual(out["value"][0], "released")

    def test_broken_switch_never_holds_a_job_back(self):
        def _boom():
            raise RuntimeError("settings unreadable")
        box = ["turn"]
        g = _strict_gate(box, strict=_boom)
        self.assertFalse(g.strict())
        th, out = _run_in_thread(_bg_slot, g, "chappie", opt_in=True)
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(out["value"][0], "shadow")

    def test_the_switch_never_touches_the_first_tagged_jobs(self):
        # learn_from_turn & co. are not opt-in: switch off, they still wait.
        box = ["turn"]
        g = _strict_gate(box, strict=lambda: False)
        th, out = _run_in_thread(_bg_slot, g, "learn_from_turn")
        time.sleep(0.15)
        self.assertTrue(th.is_alive())
        box[0] = None
        th.join(5)
        self.assertEqual(out["value"][0], "released")

    def test_nested_opt_in_keeps_the_outer_strict_job(self):
        box = ["turn"]
        g = _strict_gate(box, strict=lambda: False)

        def _nested():
            with lt.background_work("session-checkpoint"):
                with lt.background_work("chappie", opt_in=True) as job:
                    with lt.slot(g) as p:
                        return job.tag, job.opt_in, p.outcome
        th, out = _run_in_thread(_nested)
        time.sleep(0.15)
        self.assertTrue(th.is_alive(), "the inner opt-in tag unheld the job")
        box[0] = None
        th.join(5)
        self.assertEqual(out["value"], ("session-checkpoint", False,
                                        "released"))

    def test_untagged_and_main_thread_unaffected(self):
        box = ["turn"]
        g = _strict_gate(box, strict=lambda: True)
        with lt.background_work("chappie", opt_in=True):
            with lt.slot(g) as p:          # main thread
                self.assertIsNone(p)

        def _owner():
            with lt.slot(g) as p:
                return p
        th, out = _run_in_thread(_owner)
        th.join(5)
        self.assertFalse(th.is_alive(), "an owner call waited")
        self.assertIsNone(out["value"])

    def test_wait_for_quiet_reports_shadow(self):
        box = ["turn"]
        g = _strict_gate(box, strict=lambda: False)

        def _bg():
            with lt.background_work("credits-monitor", opt_in=True):
                return lt.wait_for_quiet(g)
        th, out = _run_in_thread(_bg)
        th.join(5)
        self.assertEqual(out["value"], "shadow")

    def test_configure_installs_the_switch(self):
        g = lt.BackgroundGate(log=lambda m: None)
        self.assertTrue(g.strict())
        g.configure(strict=lambda: False)
        self.assertFalse(g.strict())
        g.configure(strict=None)            # None leaves it as it is
        self.assertFalse(g.strict())


class JobCapTests(unittest.TestCase):
    """max_defer_s gives ONE job a shorter soft cap; never a longer one, and
    the hard grace still bounds a turn in progress."""

    def test_clean_cap(self):
        for bad in (None, 0, 0.0, -1, "x", True, False, object()):
            self.assertIsNone(lt._clean_cap(bad), bad)
        self.assertEqual(lt._clean_cap(15), 15.0)
        self.assertEqual(lt._clean_cap("2.5"), 2.5)

    def test_job_cap_shortens_a_quiet_window_wait(self):
        box = ["conversation"]
        g = _strict_gate(box, strict=lambda: True, cap=60.0, grace=60.0)
        t0 = time.monotonic()
        th, out = _run_in_thread(_bg_slot, g, "notification-triage",
                                 opt_in=True, max_defer_s=0.2)
        th.join(5)
        self.assertFalse(th.is_alive(), "the job's own cap was ignored")
        self.assertEqual(out["value"][0], "forced")
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_job_cap_still_waits_out_a_turn_up_to_the_hard_grace(self):
        box = ["turn"]
        g = _strict_gate(box, strict=lambda: True, cap=60.0, grace=0.4)
        t0 = time.monotonic()
        th, out = _run_in_thread(_bg_slot, g, "notification-triage",
                                 opt_in=True, max_defer_s=0.1)
        time.sleep(0.25)
        self.assertTrue(th.is_alive(), "forced into the middle of the turn")
        th.join(5)
        self.assertFalse(th.is_alive(), "unbounded wait")
        self.assertEqual(out["value"][0], "forced")
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_job_cap_never_lengthens_the_gate_cap(self):
        box = ["conversation"]
        g = _strict_gate(box, strict=lambda: True, cap=0.2, grace=60.0)
        th, out = _run_in_thread(_bg_slot, g, "x", opt_in=True,
                                 max_defer_s=60.0)
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(out["value"][0], "forced")

    def test_acquire_takes_the_job_cap(self):
        box = ["conversation"]
        g = _gate(box, cap=60.0, grace=60.0)
        th, out = _run_in_thread(g.acquire, "x", max_defer_s=0.2)
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(out["value"].outcome, "forced")


if __name__ == "__main__":
    unittest.main()
