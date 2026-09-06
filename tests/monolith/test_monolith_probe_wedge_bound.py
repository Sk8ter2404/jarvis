"""``_probe_camera_index`` must not manufacture wedges, and the wedges it does
report must be BOUNDED BY THE CODE.

THE DEFECT THESE TESTS PIN (measured on the owner's rig 2026-09-06, index 0
held open by a second process, nothing stubbed):

    1. THE ARITHMETIC MANUFACTURED THE DIAGNOSIS. The worker started its
       read-until-deadline clock AFTER the open returned, so its real cost was
       ``open + (timeout_sec - 0.2)`` while its joiner waited only
       ``timeout_sec + 0.5``. Any open slower than 0.7 s therefore overran the
       joiner by construction:

           open 0.936 s -> worker_total 2.740 s   joiner budget 2.500 s  OVER
           open 1.053 s -> worker_total 2.856 s                          OVER
           open 1.189 s -> worker_total 2.990 s                          OVER
           open 1.441 s -> worker_total 3.249 s                          OVER

       and the joiner's overrun branch does not merely give up: it RETIRES the
       process-wide camera I/O lock and announces "wedged inside its camera
       open". Eight seconds after the last of those the process was back at its
       baseline thread count - nothing was wedged and nothing was leaked.

    2. NOTHING BOUNDED THE COUNT. The producer's bounded opener has scored a
       quarantine strike on its retirement branch since 2026-09-05; this call
       site scored nothing and gated nothing. Soaked at the boot cadence with
       the camera left in place (one probe every 6 s), it retired ONE LOCK PER
       ATTEMPT for the whole run - 219 retirements in 22 minutes, no ceiling.

    3. THE OBSERVED "BOUND" WAS A DELETED WORKLOAD. A boot log showing
       "3 retirements at 04:31:02/08/14, then stable for 17 minutes" was read
       as evidence of a bound. It is not: boot probes each camera exactly three
       times (first pass, retry pass, by-name rescue) and then preflight DROPS
       the index. Nothing retried it, so nothing could be retired. The counter
       did not stabilise; the workload was deleted.

Every test below fails on the pre-fix tree and passes on the fixed one. Nothing
here opens a real camera: the "opener" is a stub and the "capture" is a plain
object with ``read()`` and ``release()``.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _NoFrameCap:
    """A capture handle that opens fine and never produces an image — the
    single most common real-world shape (a device another process is holding),
    and the one the budget bug turned into a "wedge"."""

    def __init__(self):
        self.released = False
        self.reads = 0

    def read(self):
        self.reads += 1
        return False, None

    def release(self):
        self.released = True


class _ProbeWedgeTestBase(MonolithGlobalsTestCase):
    """Own lock + own quarantine registry per test: a retirement is permanent
    by design and a bench outlives the call, so neither may leak sideways."""

    def setUp(self):
        bc = self.bc
        self.lock = bc._CameraIOLock()
        p = mock.patch.object(bc, "_camera_io_lock", self.lock)
        p.start()
        self.addCleanup(p.stop)
        with bc._camera_quarantine_lock:
            bc._camera_quarantine.clear()
        self._wedges: list[threading.Event] = []
        self.addCleanup(self._release_every_wedge)

    def _release_every_wedge(self):
        for ev in self._wedges:
            ev.set()
        # Give the freed workers a moment to leave the lock they are holding so
        # the next test does not start against a stub still inside a `with`.
        time.sleep(0.15)

    # ---- stub openers -----------------------------------------------------
    def _slow_open_no_frame(self, open_delay: float):
        """The MEASURED failure: the open itself is slow (0.9-1.4 s on this
        rig) and the device then never yields a frame. Returns (opener, caps)."""
        caps: list[_NoFrameCap] = []

        def _open(idx, *a, **kw):
            time.sleep(open_delay)
            cap = _NoFrameCap()
            caps.append(cap)
            return cap

        return _open, caps

    def _never_returns(self):
        """A REAL wedge: the open never comes back on its own. The worker is
        parked inside the camera I/O lock exactly as a wedged driver call is."""
        ev = threading.Event()
        self._wedges.append(ev)
        calls: list[int] = []

        def _open(idx, *a, **kw):
            calls.append(1)
            ev.wait(30.0)
            return None

        return _open, calls


@requires_monolith
class SlowOpenIsNotAWedgeTests(_ProbeWedgeTestBase):
    """Symptom 1: a camera that opens slowly and yields nothing is a camera
    that opens slowly and yields nothing — not a wedged driver call, and not a
    reason to retire the process-wide lock."""

    def test_a_slow_open_with_no_frame_retires_nothing(self):
        """PRE-FIX: one retirement, every single time. The worker needed
        open + (timeout-0.2) = 1.70 s of a 1.50 s joiner budget."""
        opener, _caps = self._slow_open_no_frame(0.9)
        with mock.patch.object(self.bc, "_camera_open", opener):
            ok = self.bc._probe_camera_index(0, timeout_sec=1.0)
        self.assertFalse(ok, "a camera that never yields a frame is not usable")
        self.assertEqual(
            self.lock.retired_count(), 0,
            "a slow open was reported as a wedged driver call and cost the "
            "process-wide camera I/O lock a retirement")

    def test_a_slow_open_with_no_frame_scores_no_quarantine_strike(self):
        """A manufactured wedge would also bench a camera that never wedged."""
        opener, _caps = self._slow_open_no_frame(0.9)
        with mock.patch.object(self.bc, "_camera_open", opener):
            self.bc._probe_camera_index(0, timeout_sec=1.0)
        self.assertNotIn(0, self.bc.get_camera_quarantine(),
                         "a slow open scored a strike it did not earn")

    def test_the_worker_finishes_inside_the_joiners_budget(self):
        """The invariant behind the fix, stated directly: the worker's whole
        cost (open + reads) must fit in what the joiner agreed to wait."""
        opener, caps = self._slow_open_no_frame(0.9)
        t0 = time.monotonic()
        with mock.patch.object(self.bc, "_camera_open", opener):
            self.bc._probe_camera_index(0, timeout_sec=1.0)
        elapsed = time.monotonic() - t0
        self.assertLess(
            elapsed, 1.5,
            f"the probe took {elapsed:.2f}s — it burned the whole joiner "
            f"budget, which is what the abandon-and-retire branch reads as a "
            f"wedge")
        self.assertTrue(caps and caps[0].released,
                        "the capture handle was never released")

    def test_a_fast_camera_still_gets_its_warm_up_window(self):
        """The fix must not shorten the warm-up a healthy-but-warming camera
        depends on: with an instant open the read budget is unchanged."""
        caps: list[_NoFrameCap] = []

        def _open(idx, *a, **kw):
            cap = _NoFrameCap()
            caps.append(cap)
            return cap

        with mock.patch.object(self.bc, "_camera_open", _open):
            t0 = time.monotonic()
            self.bc._probe_camera_index(0, timeout_sec=1.0)
            elapsed = time.monotonic() - t0
        self.assertGreater(
            elapsed, 0.7,
            "the probe gave up early — a camera that returns False frames for "
            "its first ~2 s (measured on the Logi) would be failed on sight")
        self.assertGreaterEqual(caps[0].reads, 2,
                                "the read loop did not retry at all")


@requires_monolith
class RealWedgeIsStillReportedTests(_ProbeWedgeTestBase):
    """The fix must not throw away the real signal with the false one."""

    def test_an_open_that_never_returns_still_retires_the_lock(self):
        opener, calls = self._never_returns()
        with mock.patch.object(self.bc, "_camera_open", opener):
            ok = self.bc._probe_camera_index(0, timeout_sec=0.3)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 1, "the stub opener never ran")
        self.assertEqual(self.lock.retired_count(), 1,
                         "a genuinely wedged open no longer retires the "
                         "poisoned lock — every later camera op would queue "
                         "on a corpse")

    def test_a_real_wedge_scores_a_quarantine_strike(self):
        """PRE-FIX: this call site retired the process-wide lock and told the
        quarantine registry NOTHING, so nothing counted probe wedges and
        nothing could ever stop them."""
        opener, _calls = self._never_returns()
        with mock.patch.object(self.bc, "_camera_open", opener):
            self.bc._probe_camera_index(0, timeout_sec=0.3)
        entry = self.bc.get_camera_quarantine().get(0)
        self.assertIsNotNone(entry, "the wedged index scored no strike at all")
        self.assertEqual(entry["quarantine_strikes"], 1)


@requires_monolith
class RetirementsAreBoundedByTheCodeTests(_ProbeWedgeTestBase):
    """Symptom 2 + 3, and the whole point: the count must stop because the
    CODE stops it, not because the caller ran out of calls."""

    def _hammer(self, attempts: int, timeout: float = 0.3):
        opener, calls = self._never_returns()
        with mock.patch.object(self.bc, "_camera_open", opener):
            for _ in range(attempts):
                self.bc._probe_camera_index(0, timeout_sec=timeout)
        return calls

    def test_a_permanently_wedging_index_stops_costing_retirements(self):
        """PRE-FIX: retirements == attempts, forever (soaked live at the boot
        cadence: 219 retirements in 22 minutes and still climbing)."""
        strikes = self.bc._CAMERA_QUARANTINE_STRIKES
        attempts = strikes * 4
        self._hammer(attempts)
        self.assertEqual(
            self.lock.retired_count(), strikes,
            f"{attempts} attempts against one permanently wedged index "
            f"retired {self.lock.retired_count()} locks; the bench should have "
            f"capped it at {strikes}")

    def test_the_index_is_benched_once_the_cap_is_reached(self):
        self._hammer(self.bc._CAMERA_QUARANTINE_STRIKES)
        self.assertTrue(
            self.bc._camera_is_quarantined(0),
            "the index that wedged every attempt was never benched, so the "
            "next caller starts another abandoned worker for no new "
            "information")

    def test_a_benched_index_is_not_opened_at_all(self):
        """The bench has to mean 'do not start a worker', not merely 'expect
        it to fail' — starting one is the entire cost being bounded."""
        strikes = self.bc._CAMERA_QUARANTINE_STRIKES
        calls = self._hammer(strikes * 3)
        self.assertEqual(
            len(calls), strikes,
            f"the opener body ran {len(calls)} times for {strikes * 3} "
            f"attempts — a benched index is still being opened")

    def test_a_benched_index_fails_fast_instead_of_burning_the_budget(self):
        self._hammer(self.bc._CAMERA_QUARANTINE_STRIKES)
        opener, _calls = self._never_returns()
        t0 = time.monotonic()
        with mock.patch.object(self.bc, "_camera_open", opener):
            ok = self.bc._probe_camera_index(0, timeout_sec=0.3)
        elapsed = time.monotonic() - t0
        self.assertFalse(ok)
        self.assertLess(elapsed, 0.2,
                        f"a benched index still cost {elapsed:.2f}s of joiner "
                        f"budget")

    def test_the_bench_expires_so_a_merely_busy_camera_self_heals(self):
        """Bounded must not mean permanently dead: an index benched for a
        transient wedge re-tests itself once the backoff elapses."""
        self._hammer(self.bc._CAMERA_QUARANTINE_STRIKES)
        self.assertTrue(self.bc._camera_is_quarantined(0))
        with self.bc._camera_quarantine_lock:
            # Fast-forward past the bench rather than sleeping 60 s for it.
            self.bc._camera_quarantine[0]["until"] = time.time() - 1.0
        caps: list[_NoFrameCap] = []

        def _healthy(idx, *a, **kw):
            cap = _NoFrameCap()
            cap.read = lambda: (True, object())      # a real frame arrives
            caps.append(cap)
            return cap

        with mock.patch.object(self.bc, "_camera_open", _healthy):
            ok = self.bc._probe_camera_index(0, timeout_sec=0.3)
        self.assertTrue(ok, "the index never got another chance after its "
                            "bench expired")
        self.assertEqual(len(caps), 1)


@requires_monolith
class TheRetirementMessageSaysOnlyWhatIsKnownTests(_ProbeWedgeTestBase):
    """Symptom 3's other half. The announcement used to assert that the
    abandoned worker 'keeps the old lock forever'. That was never measured, and
    it is false for the case that produces most retirements — the worker came
    back 0.2-0.8 s later and the process returned to its baseline thread count
    within 8 s. An external reviewer read that sentence as proof that every
    retirement permanently leaks a worker thread and built a capacity argument
    on top of it."""

    def test_the_announcement_does_not_claim_a_permanent_leak(self):
        with self.assertLogs(level="WARNING") as cm:
            self.lock.retire("unit test")
        joined = "\n".join(cm.output)
        self.assertIn("camera I/O lock was RETIRED", joined)
        self.assertNotIn(
            "keeps the old lock forever", joined,
            "the retirement line still asserts a permanent thread leak it has "
            "never measured")

    def test_the_announcement_still_names_the_cumulative_count(self):
        with self.assertLogs(level="WARNING") as cm:
            self.lock.retire("unit test")
        self.assertIn("Retired locks this session: 1.", "\n".join(cm.output))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
