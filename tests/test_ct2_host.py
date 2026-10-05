"""core/ct2_host.py - every CTranslate2 call on a CUDA model runs on ONE
thread that never exits (v2.0.180).

THE CRASH (v2.0.179, 2026-10-04 20:40, crash dumps): the self-diagnostic's
throwaway ``probe-stt`` thread decoded on cuda:1 and EXITED; CTranslate2's
per-thread CUDA destructors ran in the loader's thread teardown, threw (no
CUDA context) and the process aborted. A thread that exits must never own
CTranslate2 CUDA state.

Pinned here with a fake model that records which threads would own that
state (no GPU, no ctranslate2):
  * run() executes on the single, persistent ct2-host thread - never on the
    short-lived caller - and the host survives its callers;
  * decode() runs transcribe() AND the drain of its lazy generator (where
    faster-whisper really decodes) on the host for a CUDA device, inline for
    the CPU;
  * exceptions reach the caller with their type, and the host clears the
    frames' locals first (no CUDA object rides a traceback to a dying thread);
  * a call made from inside a host job runs inline (no self-deadlock);
  * two callers' jobs never overlap;
  * retire() releases a dropped model ON the host.

    python -m unittest tests.test_ct2_host
"""
from __future__ import annotations

import gc
import threading
import types
import unittest
import weakref
from unittest import mock

from core import ct2_host


class FakeCT2Whisper:
    """faster-whisper's surface: transcribe() returns (lazy generator, info).
    Records the thread of every call that would leave CTranslate2 CUDA state
    on its caller - transcribe() itself and the generator body."""

    def __init__(self):
        self.state_threads: list = []

    def _touch(self):
        self.state_threads.append(threading.current_thread())

    def transcribe(self, audio, **kw):
        self._touch()

        def _gen():
            self._touch()
            yield types.SimpleNamespace(text="hi", no_speech_prob=0.1,
                                        avg_logprob=-0.2)
        return _gen(), types.SimpleNamespace(no_speech_prob=0.1)


def _in_short_lived_thread(fn, name="probe-stt"):
    box = {}

    def _run():
        try:
            box["result"] = fn()
        except BaseException as e:      # handed back to the test
            box["exc"] = e
        box["thread"] = threading.current_thread()
    t = threading.Thread(target=_run, name=name, daemon=True)
    t.start()
    t.join(10.0)
    if t.is_alive():
        raise AssertionError("the short-lived caller did not finish")
    return box


class RunTests(unittest.TestCase):
    def test_runs_on_one_persistent_host_thread_not_the_caller(self):
        seen = []
        box1 = _in_short_lived_thread(
            lambda: ct2_host.run(lambda: seen.append(threading.current_thread())))
        box2 = _in_short_lived_thread(
            lambda: ct2_host.run(lambda: seen.append(threading.current_thread())))
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], seen[1])            # one host, reused
        self.assertEqual(seen[0].name, "ct2-host")
        self.assertIsNot(seen[0], box1["thread"])  # never the caller
        self.assertIsNot(seen[0], box2["thread"])
        self.assertFalse(box1["thread"].is_alive())
        self.assertTrue(seen[0].is_alive())        # the host outlives them
        self.assertTrue(seen[0].daemon)

    def test_returns_the_result_and_passes_device_kwargs_through(self):
        def _ctor(name, device=None, device_index=None, compute_type=None):
            return (name, device, device_index, compute_type)
        got = ct2_host.run_for("cuda", _ctor, "large-v3-turbo", device="cuda",
                               device_index=1, compute_type="int8")
        self.assertEqual(got, ("large-v3-turbo", "cuda", 1, "int8"))

    def test_exception_reaches_the_caller_with_its_type(self):
        def _boom():
            raise RuntimeError("CUDA failed with error out of memory")
        with self.assertRaises(RuntimeError) as cm:
            ct2_host.run(_boom)
        self.assertIn("out of memory", str(cm.exception))

    def test_host_clears_frame_locals_before_handing_back_an_exception(self):
        class _CudaObject:
            pass

        def _job():
            obj = _CudaObject()           # a local the traceback would pin
            box["ref"] = weakref.ref(obj)
            raise RuntimeError("boom")
        box = {}
        try:
            ct2_host.run(_job)
        except RuntimeError as e:
            caught = e
        gc.collect()
        self.assertIsNotNone(caught.__traceback__)
        self.assertIsNone(box["ref"]())   # freed although we still hold `caught`

    def test_nested_call_from_a_host_job_runs_inline(self):
        inner = []

        def _outer():
            return ct2_host.run(lambda: inner.append(threading.current_thread())
                                or "inner-ok")
        self.assertEqual(ct2_host.run(_outer), "inner-ok")
        self.assertIs(inner[0], ct2_host.host_thread())

    def test_two_callers_never_overlap(self):
        active = [0]
        peak = [0]
        lock = threading.Lock()
        release = threading.Event()

        def _job():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            release.wait(0.2)
            with lock:
                active[0] -= 1
        ts = [threading.Thread(target=ct2_host.run, args=(_job,), daemon=True)
              for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10.0)
        self.assertEqual(peak[0], 1)

    def test_the_host_survives_an_escape_from_its_own_loop(self):
        # The host's own exit is the v2.0.179 abort: nothing that escapes a
        # loop pass (here a SystemExit out of the queue read) may end it, and
        # a job queued next is still served by the SAME thread.
        ct2_host.run(lambda: None)
        host = ct2_host.host_thread()
        real_q = ct2_host._q

        class _FlakyQ:
            boom = 1

            def get(self, timeout=None):
                if _FlakyQ.boom:
                    _FlakyQ.boom = 0
                    raise SystemExit("escaped the host loop")
                return real_q.get(timeout=timeout)

            def put(self, item):
                real_q.put(item)

            def empty(self):
                return real_q.empty()
        with mock.patch.object(ct2_host, "_q", _FlakyQ()):
            ran = []
            ct2_host.run(lambda: ran.append(threading.current_thread()))
            ct2_host.run(lambda: ran.append(threading.current_thread()))
        self.assertEqual(_FlakyQ.boom, 0)       # the escape really happened
        self.assertEqual(ran, [host, host])
        self.assertTrue(host.is_alive())
        self.assertIs(ct2_host.host_thread(), host)

    def test_run_for_cpu_is_inline(self):
        seen = []
        ct2_host.run_for("cpu", lambda: seen.append(threading.current_thread()))
        self.assertIs(seen[0], threading.current_thread())

    def test_is_cuda(self):
        self.assertTrue(ct2_host.is_cuda("cuda"))
        self.assertTrue(ct2_host.is_cuda("cuda:1"))
        self.assertFalse(ct2_host.is_cuda("cpu"))
        self.assertFalse(ct2_host.is_cuda(None))


class DecodeTests(unittest.TestCase):
    def test_cuda_decode_and_generator_drain_never_touch_the_caller(self):
        # THE 179 PATH: a short-lived probe thread decodes on cuda:1.
        model = FakeCT2Whisper()
        box = _in_short_lived_thread(
            lambda: ct2_host.decode(model, [0.0] * 16, "cuda:1", language="en"))
        segs, info = box["result"]
        self.assertEqual([s.text for s in segs], ["hi"])
        self.assertEqual(len(model.state_threads), 2)   # transcribe + drain
        for t in model.state_threads:
            self.assertEqual(t.name, "ct2-host")
            self.assertIsNot(t, box["thread"])

    def test_cpu_decode_stays_inline(self):
        model = FakeCT2Whisper()
        segs, _info = ct2_host.decode(model, [0.0] * 16, "cpu", language="en")
        self.assertEqual(len(segs), 1)
        self.assertEqual(model.state_threads,
                         [threading.current_thread()] * 2)

    def test_a_cuda_model_is_hosted_even_when_the_label_says_cpu(self):
        # Review 2026-10-04: the probe reads _stt and _stt_device apart,
        # outside the STT lock - a stale "cpu" label must not send a CUDA
        # model's decode to the probe's own (exiting) thread.
        model = FakeCT2Whisper()
        model.model = types.SimpleNamespace(device="cuda")
        box = _in_short_lived_thread(
            lambda: ct2_host.decode(model, [0.0] * 16, "cpu", language="en"))
        self.assertEqual(len(box["result"][0]), 1)
        self.assertEqual({t.name for t in model.state_threads}, {"ct2-host"})

    def test_a_cpu_model_with_a_cpu_label_stays_inline(self):
        model = FakeCT2Whisper()
        model.model = types.SimpleNamespace(device="cpu")
        ct2_host.decode(model, [0.0] * 16, "cpu", language="en")
        self.assertEqual(model.state_threads, [threading.current_thread()] * 2)

    def test_model_device_never_raises(self):
        class _Boom:
            @property
            def model(self):
                raise RuntimeError("boom")
        self.assertEqual(ct2_host.model_device(_Boom()), "")
        self.assertEqual(ct2_host.model_device(mock.Mock()), "")
        self.assertEqual(ct2_host.model_device(None), "")
        self.assertEqual(ct2_host.model_device(
            types.SimpleNamespace(model=types.SimpleNamespace(device="CUDA"))),
            "cuda")

    def test_generator_is_closed_when_the_drain_raises(self):
        closed = []

        class _Model:
            def transcribe(self, audio, **kw):
                def _gen():
                    try:
                        yield types.SimpleNamespace(text="a")
                        raise RuntimeError("CUDA failed mid-decode")
                    finally:
                        closed.append(threading.current_thread())
                return _gen(), None
        with self.assertRaises(RuntimeError):
            ct2_host.decode(_Model(), [0.0], "cuda")
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].name, "ct2-host")


class RetireTests(unittest.TestCase):
    def test_retired_model_is_released_on_the_host(self):
        freed_on = []

        class _Model:
            pass
        m = _Model()
        weakref.finalize(m, lambda: freed_on.append(threading.current_thread()))
        # A short grace: long enough for `del m` below to run first (the
        # caller drops its reference right after retire), short for a test.
        with mock.patch.object(ct2_host, "_RETIRE_GRACE_S", 0.2):
            ct2_host.retire(m)
            del m
            done = threading.Event()
            for _ in range(80):
                if freed_on:
                    break
                ct2_host.run(lambda: None)      # nudge the host loop
                done.wait(0.05)
        self.assertEqual(len(freed_on), 1)
        self.assertEqual(freed_on[0].name, "ct2-host")

    def _nudge_until(self, cond, rounds=80):
        for _ in range(rounds):
            if cond():
                return True
            ct2_host.run(lambda: None)          # nudge the host loop
            threading.Event().wait(0.05)
        return cond()

    def test_another_holder_never_frees_it_on_its_own_thread(self):
        # Review 2026-10-04: a probe that read _stt just before the drop
        # still holds the model. The host must keep its reference until the
        # probe lets go, so the LAST reference - and CTranslate2's teardown,
        # which frees CUDA memory through a per-thread stream - goes on the
        # host, never on the probe's exiting thread.
        freed_on = []

        class _Model:
            pass
        m = _Model()
        weakref.finalize(m, lambda: freed_on.append(threading.current_thread()))
        holder = [m]
        released0 = ct2_host.stats()["released"]
        with mock.patch.object(ct2_host, "_RETIRE_GRACE_S", 0.1):
            ct2_host.retire(m)
            del m
            # Well past the grace: still held by the host, nothing freed.
            self.assertFalse(self._nudge_until(lambda: bool(freed_on), rounds=10))
            self.assertEqual(ct2_host.stats()["released"], released0)
            _in_short_lived_thread(holder.clear, name="probe-stt")
            self.assertEqual(freed_on, [])      # not on the probe's thread
            self.assertTrue(self._nudge_until(lambda: bool(freed_on)))
        self.assertEqual(freed_on[0].name, "ct2-host")

    def test_a_dropped_model_is_released_before_the_next_job_runs(self):
        # The job after a CUDA fault is typically the replacement's build:
        # the dead model's VRAM (on a 4 GB 1650) must be free before it runs,
        # not 2 s later (the original fixed grace).
        freed = []

        class _Model:
            pass
        m = _Model()
        weakref.finalize(m, lambda: freed.append(threading.current_thread()))
        ct2_host.retire(m)
        del m
        seen_by_next_job = ct2_host.run(lambda: list(freed))
        self.assertEqual(len(seen_by_next_job), 1)
        self.assertEqual(seen_by_next_job[0].name, "ct2-host")

    def test_a_holder_that_never_lets_go_is_given_up_after_the_max_hold(self):
        class _Model:
            pass
        m = _Model()
        holder = [m]
        released0 = ct2_host.stats()["released"]
        with mock.patch.object(ct2_host, "_RETIRE_GRACE_S", 0.05), \
                mock.patch.object(ct2_host, "_RETIRE_MAX_HOLD_S", 0.3):
            ct2_host.retire(m)
            del m
            self.assertTrue(self._nudge_until(
                lambda: ct2_host.stats()["released"] > released0))
        self.assertEqual(len(holder), 1)        # still alive: freed as before

    def test_retire_none_is_a_no_op(self):
        before = ct2_host.stats()["retired"]
        ct2_host.retire(None)
        self.assertEqual(ct2_host.stats()["retired"], before)


if __name__ == "__main__":
    unittest.main()
