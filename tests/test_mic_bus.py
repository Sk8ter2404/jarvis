"""The always-open microphone (MIC_BUS_MODE, 2026-10-05): core/mic_bus.py.

A fake stream stands in for PortAudio: open_stream hands back an object the
test drives by calling the bus's REAL callback, exactly as PortAudio would.
No device is opened. numpy + stdlib (light tier).

    python -m unittest tests.test_mic_bus
"""
from __future__ import annotations

import queue
import threading
import time
import unittest

import numpy as np

from core import mic_bus as mb

SR = 16000


class FakeStream:
    def __init__(self, device, cb):
        self.device, self.cb = device, cb
        self.closed = False

    def feed(self, x):
        x = np.asarray(x, np.float32).reshape(-1, 1)
        self.cb(x, len(x), None, None)


class Harness:
    """A bus with fakes for every injected dependency."""

    def __init__(self, **kw):
        self.opened = []
        self.closed = []
        self.owner = [False]
        self.run = [True]
        self.fail = [None]
        self.taps = []
        self.tap_ok = [True]
        self.reopens = [0]

        def open_stream(device, cb):
            if self.fail[0] is not None:
                raise self.fail[0]
            st = FakeStream(device, cb)
            self.opened.append(st)
            return st

        def close_stream(st):
            st.closed = True
            self.closed.append(st)

        def claim():
            if self.owner[0]:
                return False
            self.owner[0] = True
            return True

        def release():
            self.owner[0] = False

        self.bus = mb.MicBus(
            open_stream=open_stream, close_stream=close_stream, claim=claim,
            release=release, should_run=lambda: self.run[0],
            tap_fanout=lambda fr: self.taps.append(fr.lin),
            tap_allowed=lambda: self.tap_ok[0],
            on_reopen=lambda: self.reopens.__setitem__(0, self.reopens[0] + 1),
            log=lambda line: None, **kw)

    def close(self):
        self.bus.shutdown()

    def stream(self):
        return self.opened[-1]

    def wait(self, cond, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False


def chunk(value, n=1024):
    return np.full(n, value, np.float32)


class OpenCloseTests(unittest.TestCase):

    def test_ensure_opens_once_and_stays_open(self):
        h = Harness()
        self.addCleanup(h.close)
        ok, err = h.bus.ensure(3)
        self.assertTrue(ok, err)
        self.assertEqual(len(h.opened), 1)
        self.assertEqual(h.stream().device, 3)
        self.assertTrue(h.owner[0])            # the PortAudio owner cell
        ok, _ = h.bus.ensure(3)                # the next capture: no reopen
        self.assertTrue(ok)
        self.assertEqual(len(h.opened), 1)
        self.assertEqual(h.reopens[0], 1)      # the canceller re-anchors once

    def test_a_new_device_reopens(self):
        h = Harness()
        self.addCleanup(h.close)
        h.bus.ensure(3)
        ok, _ = h.bus.ensure(5)
        self.assertTrue(ok)
        self.assertEqual([s.device for s in h.opened], [3, 5])
        self.assertTrue(h.opened[0].closed)

    def test_mute_closes_the_stream_and_releases_the_owner(self):
        h = Harness()
        self.addCleanup(h.close)
        h.bus.ensure(3)
        h.run[0] = False                       # the tray's mic-pause
        h.bus.wake()
        self.assertTrue(h.wait(lambda: not h.bus.is_open()))
        self.assertTrue(h.stream().closed)
        self.assertFalse(h.owner[0])
        ok, err = h.bus.ensure(3)              # muted: nothing opens
        self.assertFalse(ok)
        self.assertIsInstance(err, mb.BusUnavailable)

    def test_a_failed_open_is_reported_not_retried(self):
        h = Harness()
        self.addCleanup(h.close)
        h.fail[0] = OSError("device gone")
        ok, err = h.bus.ensure(3)
        self.assertFalse(ok)
        self.assertIsInstance(err, OSError)
        self.assertNotIsInstance(err, mb.BusUnavailable)
        self.assertFalse(h.owner[0])           # released after the failure
        time.sleep(0.3)
        self.assertEqual(len(h.opened), 0)     # no retry of its own (R10)

    def test_suspend_closes_and_resume_lets_it_reopen(self):
        h = Harness()
        self.addCleanup(h.close)
        h.bus.ensure(3)
        self.assertTrue(h.bus.suspend())
        self.assertFalse(h.bus.is_open())
        self.assertFalse(h.owner[0])
        ok, err = h.bus.ensure(3)
        self.assertFalse(ok)
        self.assertIsInstance(err, mb.BusUnavailable)
        h.bus.resume()
        ok, _ = h.bus.ensure(3)
        self.assertTrue(ok)

    def test_a_dead_stream_is_closed(self):
        h = Harness()
        self.addCleanup(h.close)
        old = mb.DEAD_AFTER_S
        mb.DEAD_AFTER_S = 0.2
        self.addCleanup(setattr, mb, "DEAD_AFTER_S", old)
        h.bus.ensure(3)
        self.assertTrue(h.wait(lambda: not h.bus.is_open()))
        self.assertTrue(h.stream().closed)

    def test_the_claim_is_refused(self):
        h = Harness()
        self.addCleanup(h.close)
        h.owner[0] = True                      # a Path-B capture holds it
        ok, err = h.bus.ensure(3)
        self.assertFalse(ok)
        self.assertIsInstance(err, mb.BusUnavailable)


class FrameTests(unittest.TestCase):

    def test_frames_reach_subscribers_rings_and_taps(self):
        h = Harness()
        self.addCleanup(h.close)
        h.bus.ensure(1)
        sub = h.bus.subscribe()
        for v in (0.1, 0.2, 0.3):
            h.stream().feed(chunk(v))
        got = [sub.get(timeout=2.0) for _ in range(3)]
        self.assertEqual([f.n_end for f in got], [1024, 2048, 3072])
        self.assertAlmostEqual(float(got[1].raw[0]), 0.2, places=6)
        self.assertFalse(got[0].aec)
        np.testing.assert_array_equal(got[0].lin, got[0].raw)
        self.assertTrue(h.wait(lambda: h.bus.n_written == 3072))
        ring = h.bus.read("raw", 1024, 3072)
        self.assertAlmostEqual(float(ring[0]), 0.2, places=6)
        self.assertAlmostEqual(float(ring[-1]), 0.3, places=6)
        self.assertIsNone(h.bus.read("raw", 0, 4096))   # not written yet
        self.assertEqual(len(h.taps), 3)
        with self.assertRaises(queue.Empty):
            sub.get(timeout=0.05)               # Queue semantics
        sub.close()

    def test_a_private_capture_keeps_frames_from_the_taps(self):
        h = Harness()
        self.addCleanup(h.close)
        h.bus.ensure(1)
        sub = h.bus.subscribe()
        h.tap_ok[0] = False                    # an off-thread capture runs
        h.stream().feed(chunk(0.1))
        sub.get(timeout=2.0)
        self.assertEqual(h.taps, [])

    def test_the_canceller_runs_on_the_dsp_thread(self):
        h = Harness()
        self.addCleanup(h.close)
        seen = []

        def proc(mono, t):
            seen.append(threading.current_thread().name)
            return mono * 0.5, mono * 0.25

        h.bus.process = proc
        h.bus.ensure(1)
        sub = h.bus.subscribe()
        h.stream().feed(chunk(0.4))
        fr = sub.get(timeout=2.0)
        self.assertTrue(fr.aec)
        self.assertAlmostEqual(float(fr.lin[0]), 0.2, places=6)
        self.assertAlmostEqual(float(fr.sup[0]), 0.1, places=6)
        self.assertEqual(seen, ["mic-bus-dsp"])
        self.assertAlmostEqual(float(h.bus.read("sup", 0, 1024)[0]), 0.1,
                               places=6)

    def test_a_slow_reader_never_stalls_capture(self):
        h = Harness()
        self.addCleanup(h.close)
        h.bus.ensure(1)
        sub = h.bus.subscribe(maxsize=4)
        for i in range(20):
            h.stream().feed(chunk(i / 100.0))
        self.assertTrue(h.wait(lambda: h.bus.n_written == 20 * 1024))
        self.assertEqual(sub.dropped, 16)       # the oldest dropped
        self.assertEqual(sub.get(timeout=1.0).n_end, 17 * 1024)

    def test_the_ring_keeps_only_its_window(self):
        h = Harness(ring_s=1.0)
        self.addCleanup(h.close)
        h.bus.ensure(1)
        for i in range(40):                     # 2.56 s through a 1 s ring
            h.stream().feed(chunk(i))
        self.assertTrue(h.wait(lambda: h.bus.n_written == 40 * 1024))
        self.assertIsNone(h.bus.read("raw", 0, 1024))
        last = h.bus.last("raw", 1024)
        self.assertEqual(float(last[0]), 39.0)

    def test_dropped_frames_re_anchor_the_canceller(self):
        h = Harness()
        self.addCleanup(h.close)
        h.bus.ensure(1)
        before = h.reopens[0]
        h.bus._dsp_one(1024, 1.0, chunk(0.1))
        h.bus._dsp_one(4096, 1.2, chunk(0.1))   # 2048 samples never arrived
        self.assertEqual(h.reopens[0], before + 1)

    def test_closed_time_is_counted(self):
        now = [100.0]
        h = Harness(clock=lambda: now[0])
        self.addCleanup(h.close)
        h.bus.ensure(1)
        self.assertTrue(h.bus.suspend())
        now[0] = 104.0
        h.bus.resume()
        h.bus.ensure(1)
        self.assertAlmostEqual(h.bus.take_closed_s(), 4.0, places=3)
        self.assertEqual(h.bus.take_closed_s(), 0.0)


if __name__ == "__main__":
    unittest.main()
