"""What the PC plays, as the echo canceller's reference (2026-10-05):
core/loopback_ref.py.

The ring and its time base are driven directly; the reader thread runs
against FAKE sources (no soundcard, no pyaudiowpatch, no device). numpy +
stdlib (light tier).

    python -m unittest tests.test_loopback_ref
"""
from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from core import loopback_ref as lr

SR = 16000


class RingTests(unittest.TestCase):

    def test_write_read_and_overwrite(self):
        ref = lr.LoopbackReference(seconds=1.0, clock=lambda: 0.0)
        ref.write(np.arange(1000, dtype=np.float32), t=1.0)
        np.testing.assert_array_equal(ref.read(10, 5), [10, 11, 12, 13, 14])
        self.assertIsNone(ref.read(990, 20))          # not written yet
        ref.write(np.arange(1000, 20000, dtype=np.float32), t=2.0)
        self.assertEqual(ref.n_written, 20000)
        self.assertIsNone(ref.read(1000, 10))         # overwritten
        np.testing.assert_array_equal(ref.read(19995, 5),
                                      [19995, 19996, 19997, 19998, 19999])

    def test_time_base_takes_the_earliest_arrival(self):
        ref = lr.LoopbackReference(clock=lambda: 0.0)
        # 160-sample packets every 10 ms, arriving 0-3 ms late (jitter).
        for k, late in enumerate((0.003, 0.0, 0.002, 0.001)):
            ref.write(np.zeros(160, np.float32), t=10.0 + 0.01 * (k + 1) + late)
        # Index 640 (all four packets) was played at 10.04.
        self.assertAlmostEqual(ref.index_at(10.04), 640.0, delta=1.0)
        self.assertAlmostEqual(ref.index_at(10.0), 0.0, delta=1.0)

    def test_a_gap_restarts_the_time_base(self):
        ref = lr.LoopbackReference(clock=lambda: 0.0)
        ref.write(np.zeros(160, np.float32), t=1.0)
        seq = ref.gap_seq
        ref.mark_gap()
        self.assertEqual(ref.gap_seq, seq + 1)
        self.assertIsNone(ref.index_at(1.0))
        ref.write(np.zeros(160, np.float32), t=5.0)
        self.assertAlmostEqual(ref.index_at(5.0), 320.0, delta=0.5)

    def test_wait_for_is_bounded(self):
        ref = lr.LoopbackReference(clock=time.monotonic)
        ref._run.set()
        t0 = time.monotonic()
        self.assertFalse(ref.wait_for(100, 0.05))
        self.assertLess(time.monotonic() - t0, 1.0)
        threading.Timer(0.02, lambda: ref.write(np.zeros(200, np.float32))
                        ).start()
        self.assertTrue(ref.wait_for(100, 2.0))

    def test_rms_recent(self):
        ref = lr.LoopbackReference(clock=lambda: 0.0)
        self.assertEqual(ref.rms_recent(), 0.0)
        ref.write(np.full(SR, 0.1, np.float32), t=1.0)
        self.assertAlmostEqual(ref.rms_recent(0.5), 0.1, places=5)


class ComOrderTests(unittest.TestCase):

    def test_soundcard_is_imported_before_com_is_initialised(self):
        # soundcard initialises COM at its first import and treats
        # CoInitializeEx's S_FALSE as an error: imported AFTER our own
        # CoInitializeEx, it fails on the loopback thread. The import comes
        # first.
        calls = []
        real_import = lr.importlib.import_module

        def fake_import(name, *a, **k):
            calls.append(("import", name))
            if name == "soundcard":
                return None
            return real_import(name, *a, **k)

        class FakeOle32:
            def CoInitializeEx(self, *a):
                calls.append(("CoInitializeEx",))
                return 0

        class FakeWindll:
            ole32 = FakeOle32()

        import ctypes
        from unittest import mock
        with mock.patch.object(lr.importlib, "import_module", fake_import), (
                mock.patch.object(ctypes, "windll", FakeWindll(), create=True)):
            lr._com_init_mta()
            self.assertEqual(calls, [("import", "soundcard"),
                                     ("CoInitializeEx",)])
            calls.clear()
            lr._com_init_mta(import_soundcard=False)
            self.assertEqual(calls, [("CoInitializeEx",)])


class FakeSource:
    """A loopback source: 10 ms packets of a constant, a switchable
    default speaker, an optional failure."""
    opened = []

    def __init__(self, name="Speakers (Desk USB Audio)", key="spk1",
                 key_now=None):
        self.name, self.key = name, key
        self._key_now = key_now or (lambda: key)
        self.entered = self.exited = 0
        FakeSource.opened.append(self)

    def current_key(self):
        return self._key_now()

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False

    def read(self, n):
        time.sleep(0.002)
        return np.full((n, 2), 0.05, np.float32)


class ReaderThreadTests(unittest.TestCase):

    def setUp(self):
        FakeSource.opened = []
        self._old = lr.RESOLVE_EVERY_S
        lr.RESOLVE_EVERY_S = 0.05
        self.addCleanup(setattr, lr, "RESOLVE_EVERY_S", self._old)

    def _ref(self, sources):
        logs = []
        ref = lr.LoopbackReference(sources=sources, log=logs.append)
        self.addCleanup(ref.shutdown)
        return ref, logs

    def _wait(self, cond, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_records_downmixed_packets_into_the_ring(self):
        ref, _logs = self._ref([FakeSource])
        self.assertTrue(ref.start())
        self.assertTrue(self._wait(lambda: ref.n_written >= 1600))
        self.assertAlmostEqual(float(ref.read(0, 160)[0]), 0.05, places=6)
        self.assertEqual(ref.endpoint, "Speakers (Desk USB Audio)")

    def test_pause_closes_the_recorder(self):
        ref, _logs = self._ref([FakeSource])
        ref.start()
        self.assertTrue(self._wait(lambda: ref.n_written > 0))
        ref.pause()
        self.assertTrue(self._wait(lambda: FakeSource.opened[-1].exited == 1))
        n = ref.n_written
        time.sleep(0.1)
        self.assertEqual(ref.n_written, n)

    def test_a_moved_default_speaker_reopens_with_a_gap(self):
        state = {"key": "spk1"}
        ref, _logs = self._ref(
            [lambda: FakeSource(key=state["key"],
                                key_now=lambda: state["key"])])
        ref.start()
        self.assertTrue(self._wait(lambda: ref.opens == 1 and ref.n_written))
        seq = ref.gap_seq
        state["key"] = "spk2"
        self.assertTrue(self._wait(lambda: ref.opens >= 2))
        self.assertGreater(ref.gap_seq, seq)

    def test_failure_falls_back_then_backs_off_logging_once(self):
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise OSError("no loopback endpoint")

        old = (lr.BACKOFF_MIN_S, lr.BACKOFF_MAX_S)
        lr.BACKOFF_MIN_S, lr.BACKOFF_MAX_S = 0.01, 0.02
        self.addCleanup(lambda: (setattr(lr, "BACKOFF_MIN_S", old[0]),
                                 setattr(lr, "BACKOFF_MAX_S", old[1])))
        ref, logs = self._ref([broken, broken])
        ref.start()
        self.assertTrue(self._wait(lambda: ref.failures >= 3))
        self.assertGreaterEqual(calls["n"], 6)        # both tried each time
        self.assertEqual(sum("cannot record" in line for line in logs), 1)

    def test_the_second_source_is_the_fallback(self):
        def broken():
            raise OSError("soundcard failed")

        ref, _logs = self._ref([broken, FakeSource])
        ref.start()
        self.assertTrue(self._wait(lambda: ref.n_written > 0))
        self.assertEqual(ref.failures, 0)

    def test_nothing_records_until_started(self):
        ref, _logs = self._ref([FakeSource])
        time.sleep(0.05)
        self.assertEqual(FakeSource.opened, [])
        self.assertFalse(ref.running)


if __name__ == "__main__":
    unittest.main()
