"""The process-wide camera I/O lock must SURVIVE an abandoned worker.

THE DEFECT THESE TESTS PIN (measured 2026-09-05 with a stub opener, no camera
touched — see _CameraIOLock's own docstring for the code-side account):

``_open_capture_bounded`` gives up on a worker that does not return in time,
but that worker is parked INSIDE ``with _camera_io_lock:``. With a plain
``threading.RLock`` it therefore owned the process-wide camera lock FOREVER,
and everything downstream failed while naming the wrong culprit:

    STEP 1: wedge one open on index 0 (0.5s cap)
      -> returned None after 0.50s
    STEP 2: can ANY other thread take the camera I/O lock now?
      -> _camera_io_lock.acquire(timeout=1.0) = False
    STEP 3: open the HEALTHY primary (index 2, 'Left webcam') with a 1.0s cap
      -> returned None after 1.02s; opener body ran 0 times
    STEP 4: two more attempts on the healthy primary
      [face-track] QUARANTINE Left webcam (PRIMARY) (index 2) - 3 failed
      recovery cycles in 120s (open wedged >1.0s). Benched for 60s; its tile
      goes dark, every OTHER camera keeps publishing.
    STEP 5: guarded release of a healthy handle -> released=False

Read those together: the healthy eMeet C960 was benched for a wedge it never
had — its opener body executed ZERO times, so it never reached DirectShow at
all — and the log line announcing it claimed "every OTHER camera keeps
publishing" while no camera was publishing. One sick camera still took every
camera down; it just took three extra timeouts to get there. The
self-diagnostic's ``_camera_io_lock.acquire(timeout=2.5)`` probe returned False
forever too, so its webcam check would read UNVERIFIED for the rest of the
session.

Every test below fails on the pre-fix tree and passes on the fixed one.
Nothing here opens a real camera: the wedge is an Event and the "capture" is a
plain object with a ``release()``.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _FakeCap:
    """Minimal stand-in for a cv2.VideoCapture handle."""

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _CameraIoLockTestBase(MonolithGlobalsTestCase):
    """Gives each test its OWN _CameraIOLock so a retirement — which is
    permanent by design — cannot leak into any other test in the process."""

    def setUp(self):
        bc = self.bc
        self.lock = bc._CameraIOLock()
        patcher = mock.patch.object(bc, "_camera_io_lock", self.lock)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Every wedge in this file is released here, so no daemon worker is
        # still parked in a lock when the process moves on to the next test.
        self._wedges: list[threading.Event] = []
        self._threads: list[threading.Thread] = []
        self.addCleanup(self._unwedge_everything)

    def _unwedge_everything(self):
        for ev in self._wedges:
            ev.set()
        for t in self._threads:
            t.join(timeout=5.0)

    def _wedging_opener(self):
        """An opener with the EXACT shape of _do_open / _dshow_open: it takes
        the camera I/O lock and then never comes back out of 'DirectShow'."""
        ev = threading.Event()
        self._wedges.append(ev)

        def _opener():
            with self.bc._camera_io_lock:
                ev.wait(30.0)
                return None

        return _opener, ev

    def _counting_opener(self):
        """A HEALTHY opener. ``calls`` counts how many times its body actually
        ran — the number that exposed the bug, because the healthy camera was
        being quarantined with a body that had run zero times."""
        calls: list[int] = []

        def _opener():
            with self.bc._camera_io_lock:
                calls.append(1)
                return _FakeCap()

        return _opener, calls


@requires_monolith
class WedgedOpenDoesNotPoisonTheLockTests(_CameraIoLockTestBase):
    """The reported defect, one assertion per measured symptom."""

    def _wedge_index_zero(self, timeout=0.3):
        opener, _ev = self._wedging_opener()
        return self.bc._open_capture_bounded(
            0, opener, label="sick cam", timeout=timeout)

    def test_the_wedged_open_is_still_abandoned_and_returns_none(self):
        self.assertIsNone(self._wedge_index_zero())

    def test_another_thread_can_still_take_the_lock_after_a_wedge(self):
        """PRE-FIX: False, permanently. This is the exact call
        skills/self_diagnostic.py makes (_CameraLockHold.acquire(2.5)), so
        pre-fix its webcam probe was UNVERIFIED for the rest of the session."""
        self._wedge_index_zero()
        got = {}

        def _probe():
            got["v"] = self.bc._camera_io_lock.acquire(timeout=1.0)
            if got["v"]:
                self.bc._camera_io_lock.release()

        t = threading.Thread(target=_probe, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "the probe thread never finished")
        self.assertTrue(got.get("v"),
                        "the camera I/O lock is still poisoned by the abandoned "
                        "worker — every later open/release and the "
                        "self-diagnostic webcam probe would fail forever")

    def test_a_healthy_camera_still_opens_after_another_camera_wedges(self):
        """PRE-FIX: returned None after the full timeout with the opener body
        run 0 times — it never reached DirectShow, it only queued on a corpse."""
        self._wedge_index_zero()
        opener, calls = self._counting_opener()
        t0 = time.monotonic()
        cap = self.bc._open_capture_bounded(
            2, opener, label="Left webcam (PRIMARY)", timeout=1.0)
        elapsed = time.monotonic() - t0
        self.assertIsNotNone(cap, "the healthy primary could not be opened")
        self.assertEqual(len(calls), 1,
                         "the healthy opener's body never ran — it spent its "
                         "budget queued on the poisoned lock")
        self.assertLess(elapsed, 0.5,
                        f"the healthy open took {elapsed:.2f}s; it should not "
                        f"have waited on the wedged camera at all")

    def test_a_healthy_camera_is_never_quarantined_for_someone_elses_wedge(self):
        """PRE-FIX: three attempts printed
        'QUARANTINE Left webcam (PRIMARY) (index 2) - 3 failed recovery cycles'
        for a camera that was fine."""
        self._wedge_index_zero()
        opener, calls = self._counting_opener()
        for _ in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._open_capture_bounded(
                2, opener, label="Left webcam (PRIMARY)", timeout=1.0)
        self.assertEqual(len(calls), self.bc._CAMERA_QUARANTINE_STRIKES)
        self.assertFalse(self.bc._camera_is_quarantined(2),
                         "the HEALTHY primary was benched for a wedge that "
                         "belonged to a different camera")
        self.assertNotIn(2, self.bc.get_camera_quarantine(),
                         "the healthy primary scored strikes it did not earn")

    def test_the_camera_that_actually_wedged_is_still_struck(self):
        """The fix must not throw away the real signal with the false one."""
        self._wedge_index_zero()
        entry = self.bc.get_camera_quarantine().get(0)
        self.assertIsNotNone(entry, "the wedged camera scored no strike at all")
        self.assertEqual(entry["quarantine_strikes"], 1)

    def test_a_guarded_release_still_succeeds_after_a_wedge(self):
        """PRE-FIX: released=False, and the handle went to the deferred-release
        path because the lock could never be acquired again."""
        self._wedge_index_zero()
        cap = _FakeCap()
        self.assertTrue(
            self.bc._release_capture_guarded(cap, 2, "Left webcam (PRIMARY)",
                                             timeout=1.0))
        self.assertTrue(cap.released)

    def test_the_retirement_is_announced_loudly(self):
        """A subsystem-level recovery the owner cannot see is the bug class
        this repo exists to avoid. It must reach the log."""
        with self.assertLogs(level="WARNING") as cm:
            self._wedge_index_zero()
        joined = "\n".join(cm.output)
        self.assertIn("camera I/O lock was RETIRED", joined)
        self.assertIn("index 0", joined)
        self.assertEqual(self.lock.retired_count(), 1)


@requires_monolith
class QueuedWorkerIsNotBlamedTests(_CameraIoLockTestBase):
    """A worker that never OWNED the lock never touched DirectShow, so it
    proves nothing about its camera. Striking it is the same mistake
    _probe_camera_index's PHASE 1 comment already documents."""

    def test_a_worker_that_only_queued_scores_no_strike_and_retires_nothing(self):
        opener, calls = self._counting_opener()
        with self.bc._camera_io_lock:          # someone else is mid-operation
            cap = self.bc._open_capture_bounded(
                2, opener, label="Left webcam (PRIMARY)", timeout=0.3)
        self.assertIsNone(cap)
        self.assertEqual(calls, [], "the worker should never have reached cv2")
        self.assertNotIn(2, self.bc.get_camera_quarantine(),
                         "a camera was struck for lock contention it did not "
                         "cause and never even reached DirectShow through")
        self.assertEqual(self.lock.retired_count(), 0,
                         "a healthy, merely-busy lock was retired")

    def test_the_contention_case_says_what_actually_happened(self):
        opener, _calls = self._counting_opener()
        with self.assertLogs(level="WARNING") as cm:
            with self.bc._camera_io_lock:
                self.bc._open_capture_bounded(
                    2, opener, label="Left webcam (PRIMARY)", timeout=0.3)
        joined = "\n".join(cm.output)
        self.assertIn("without EVER being shown to reach DirectShow", joined)
        self.assertIn("No strike is scored", joined)
        self.assertNotIn("camera I/O lock was RETIRED", joined)


@requires_monolith
class RetiredLockDoesNotStrandWaitersTests(_CameraIoLockTestBase):
    """Retirement must not simply move the hang. A thread ALREADY blocked on
    the lock when it is retired has to migrate to the replacement."""

    def _park_a_wedged_owner(self):
        """Take the lock on a worker thread and hold it. Returns (event, thread)."""
        holding = threading.Event()
        ev = threading.Event()
        self._wedges.append(ev)

        def _hold():
            with self.bc._camera_io_lock:
                holding.set()
                ev.wait(30.0)

        t = threading.Thread(target=_hold, daemon=True)
        self._threads.append(t)
        t.start()
        self.assertTrue(holding.wait(5.0), "the wedge never took the lock")
        return ev, t

    def test_a_thread_already_queued_migrates_to_the_replacement_lock(self):
        self._park_a_wedged_owner()
        arrived = threading.Event()

        def _blocking_waiter():
            self.bc._camera_io_lock.acquire()   # UNBOUNDED, like the old code
            try:
                arrived.set()
            finally:
                self.bc._camera_io_lock.release()

        w = threading.Thread(target=_blocking_waiter, daemon=True)
        self._threads.append(w)
        w.start()
        time.sleep(0.2)                          # let it queue on the old lock
        self.assertFalse(arrived.is_set())
        self.lock.retire("test")
        self.assertTrue(arrived.wait(5.0),
                        "a thread queued on the retired lock was stranded — "
                        "retirement would just relocate the hang")

    def test_the_retired_lock_can_still_be_released_by_its_wedged_owner(self):
        """The wedged worker eventually unwedges and exits its ``with``. That
        must not raise: the retired lock is kept alive precisely for this."""
        ev, t = self._park_a_wedged_owner()
        self.lock.retire("test")
        ev.set()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "the wedged owner never unwound")
        # And the replacement is still usable afterwards.
        self.assertTrue(self.bc._camera_io_lock.acquire(timeout=1.0))
        self.bc._camera_io_lock.release()


@requires_monolith
class TheLockIsStillARealLockTests(_CameraIoLockTestBase):
    """Whatever else changes, this object must still serialise DirectShow
    open/release — an overlap is a 0xc0000374 heap corruption, not a glitch."""

    def test_it_still_excludes_a_second_thread(self):
        entered = threading.Event()
        result = {}

        def _other():
            result["got"] = self.bc._camera_io_lock.acquire(timeout=0.3)
            if result["got"]:
                self.bc._camera_io_lock.release()

        with self.bc._camera_io_lock:
            entered.set()
            t = threading.Thread(target=_other, daemon=True)
            t.start()
            t.join(timeout=5.0)
        self.assertFalse(result.get("got"),
                         "two threads were inside the camera I/O lock at once")

    def test_it_is_still_reentrant_for_one_thread(self):
        with self.bc._camera_io_lock:
            with self.bc._camera_io_lock:
                self.assertTrue(True)
        self.assertTrue(self.bc._camera_io_lock.acquire(timeout=1.0))
        self.bc._camera_io_lock.release()

    def test_nesting_survives_a_retirement_mid_hold(self):
        """A thread holding the OLD lock takes the NEW one for an inner
        operation; unwinding both must leave the lock free."""
        with self.bc._camera_io_lock:
            self.lock.retire("test")
            with self.bc._camera_io_lock:
                pass
        got = {}

        def _after():
            got["v"] = self.bc._camera_io_lock.acquire(timeout=1.0)
            if got["v"]:
                self.bc._camera_io_lock.release()

        t = threading.Thread(target=_after, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertTrue(got.get("v"),
                        "the lock was left held after unwinding across a "
                        "retirement")

    def test_a_late_release_waits_for_whoever_holds_the_current_lock(self):
        """The heap-corruption guard, on the helper directly."""
        cap = _FakeCap()
        released = threading.Event()

        def _late_release():
            self.bc._release_on_current_camera_lock(cap)
            released.set()

        with self.bc._camera_io_lock:            # a live camera op is running
            t = threading.Thread(target=_late_release, daemon=True)
            self._threads.append(t)
            t.start()
            time.sleep(0.3)
            self.assertFalse(cap.released,
                             "the late release overlapped a live camera "
                             "operation — that is the DirectShow heap "
                             "corruption this lock exists to prevent")
        self.assertTrue(released.wait(5.0))
        self.assertTrue(cap.released)

    def test_an_abandoned_workers_late_release_waits_too(self):
        """Same guarantee, built the way production builds it: the worker is
        abandoned, its lock is RETIRED under it, and only then does it unwedge
        and hand its capture back. That release must still be serialised
        against the replacement lock, not fired off-lock."""
        ev = threading.Event()
        self._wedges.append(ev)
        cap = _FakeCap()

        def _opener():
            with self.bc._camera_io_lock:
                ev.wait(30.0)
                return cap

        self.assertIsNone(self.bc._open_capture_bounded(
            0, _opener, label="sick cam", timeout=0.3))
        self.assertEqual(self.lock.retired_count(), 1)

        with self.bc._camera_io_lock:            # a live op on the NEW lock
            ev.set()                             # the wedge finally lets go
            time.sleep(0.4)
            self.assertFalse(cap.released,
                             "the abandoned worker released its handle while "
                             "another camera operation was in flight")
        deadline = time.monotonic() + 5.0
        while not cap.released and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(cap.released,
                        "the abandoned worker never released its handle — a "
                        "device held open until the process exits")

    def test_a_stale_hold_token_cannot_retire_a_live_lock(self):
        """"Read the owner, then retire" is a check-then-act race: the holder
        can let go and re-take the lock in the gap, and rotating the lock out
        from under a LIVE hold is the overlapping teardown, not a fix for it.
        retire_if_holding must refuse a token that has moved on."""
        second_holding = threading.Event()
        release_second = threading.Event()
        self._wedges.append(release_second)
        tokens = {}

        def _worker():
            with self.bc._camera_io_lock:        # hold #1 — observed, then gone
                tokens["first"] = self.lock.owner_hold()
            with self.bc._camera_io_lock:        # hold #2 — LIVE camera op
                second_holding.set()
                release_second.wait(20.0)

        t = threading.Thread(target=_worker, daemon=True)
        self._threads.append(t)
        t.start()
        self.assertTrue(second_holding.wait(5.0))
        self.assertEqual(
            self.lock.retire_if_holding(*tokens["first"], "stale token"), 0)
        self.assertEqual(self.lock.retired_count(), 0,
                         "the lock was rotated out from under a LIVE camera "
                         "operation on a stale observation")
        release_second.set()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
