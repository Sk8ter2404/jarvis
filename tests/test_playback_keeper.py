"""core/playback_keeper.py -- the per-turn silent speaker stream
(PLAYBACK_KEEPER, 2026-10-05).

Stdlib only (CI-light tier). No audio device is ever touched: the stream
factory, the owner-cell claim/release and the log are fakes, and every
test runs the REAL keeper thread with short timings, because the contract
being pinned is a threading one (one toucher, claim-before-open,
close-before-release, never permanent, never a hot loop).

The monolith wiring is pinned in tests/monolith/test_monolith_playback_keeper.py.

Run: python tools/run_tests.py test_playback_keeper
"""
from __future__ import annotations

import threading
import time
import unittest

from core import playback_keeper as pk


def _wait_for(pred, timeout=2.0, step=0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


class _FakeStream:
    def __init__(self, rig, device):
        self.rig = rig
        self.device = device

    def abort(self, ignore_errors=True):
        self.rig.event("abort", self.device)

    def close(self, ignore_errors=True):
        self.rig.event("close", self.device)
        block = self.rig.close_block
        if block is not None:
            block.wait(5.0)


class _Rig:
    """Fakes for open_stream / claim / release / log, recording the order
    and the thread of every call."""

    def __init__(self, claim_ok=True, open_error=None, open_delay=0.0):
        self.events = []
        self.lock = threading.Lock()
        self.claim_ok = claim_ok
        self.open_error = open_error
        self.open_delay = open_delay
        self.close_block = None
        self.cell = [False]
        self.lines = []

    def event(self, name, arg=None):
        with self.lock:
            self.events.append((name, arg, threading.get_ident()))

    def names(self):
        with self.lock:
            return [e[0] for e in self.events]

    def open_stream(self, device):
        self.event("open", device)
        if self.open_delay:
            time.sleep(self.open_delay)
        if self.open_error is not None:
            raise self.open_error
        return _FakeStream(self, device)

    def claim(self):
        self.event("claim")
        if not self.claim_ok:
            return False
        self.cell[0] = True
        return True

    def release(self):
        self.event("release")
        self.cell[0] = False

    def log(self, line):
        self.lines.append(line)

    def keeper(self, **kw):
        kw.setdefault("linger_s", 0.05)
        kw.setdefault("max_hold_s", 5.0)
        kw.setdefault("yield_s", 5.0)
        k = pk.PlaybackKeeper(open_stream=self.open_stream, claim=self.claim,
                              release=self.release, log=self.log, **kw)
        return k


class KeeperOffTests(unittest.TestCase):
    def test_disabled_keeper_holds_nothing_and_starts_no_thread(self):
        rig = _Rig()
        started = []

        def factory(**kw):
            started.append(kw)
            return threading.Thread(**kw)

        k = pk.PlaybackKeeper(open_stream=rig.open_stream, claim=rig.claim,
                              release=rig.release, log=rig.log,
                              thread_factory=factory)
        self.assertEqual(k.begin(6), 0)
        k.end(0)
        k.note_device(7)
        k.request_yield()
        self.assertTrue(k.wait_settled(0.01))
        self.assertFalse(k.is_live())
        self.assertEqual(started, [])
        self.assertEqual(rig.events, [])


class KeeperLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.rig = _Rig()
        self.k = self.rig.keeper()
        self.k.set_enabled(True)

    def tearDown(self):
        self.k.shutdown()
        if self.rig.close_block is not None:
            self.rig.close_block.set()

    def test_open_on_begin_close_after_linger_in_contract_order(self):
        caller = threading.get_ident()
        tok = self.k.begin(6)
        self.assertTrue(tok)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertTrue(self.rig.cell[0], "owner cell must be up while open")
        self.k.end(tok)
        self.assertTrue(_wait_for(lambda: "release" in self.rig.names()))
        self.assertFalse(self.k.is_live())
        self.assertFalse(self.rig.cell[0])
        # claim BEFORE open, abort + close BEFORE release.
        self.assertEqual(self.rig.names(),
                         ["claim", "open", "abort", "close", "release"])
        # Single toucher: every native call on ONE thread, not the caller's.
        native = {e[2] for e in self.rig.events
                  if e[0] in ("open", "abort", "close")}
        self.assertEqual(len(native), 1)
        self.assertNotIn(caller, native)
        self.assertEqual(self.rig.events[1][1], 6)       # opened on device 6
        self.assertTrue(any("held the speaker (device 6)" in ln
                            for ln in self.rig.lines), self.rig.lines)

    def test_lingers_between_holders_then_closes_once(self):
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)
        tok2 = self.k.begin(6)          # inside the linger: same stream
        time.sleep(0.12)                # > linger_s, but tok2 still holds
        self.assertTrue(self.k.is_live())
        self.k.end(tok2)
        self.assertTrue(_wait_for(lambda: not self.k.is_live()))
        self.assertEqual(self.rig.names().count("open"), 1)
        self.assertEqual(self.rig.names().count("close"), 1)

    def test_nested_holders(self):
        a = self.k.begin(6)
        b = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(a)
        time.sleep(0.12)
        self.assertTrue(self.k.is_live(), "b still holds the speaker")
        self.k.end(b)
        self.assertTrue(_wait_for(lambda: not self.k.is_live()))

    def test_device_change_closes_and_reopens_on_the_new_device(self):
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.note_device(7)
        self.assertFalse(self.k.is_live(), "not live on the device in use")
        self.assertTrue(_wait_for(
            lambda: [e[1] for e in self.rig.events if e[0] == "open"]
            == [6, 7]))
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)
        self.assertTrue(_wait_for(lambda: not self.rig.cell[0]))
        self.assertEqual(self.rig.names(),
                         ["claim", "open", "abort", "close", "release",
                          "claim", "open", "abort", "close", "release"])

    def test_shutdown_closes_and_latches_off(self):
        self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.shutdown()
        self.assertTrue(_wait_for(lambda: not self.rig.cell[0]))
        self.assertEqual(self.k.begin(6), 0)
        self.k.set_enabled(True)        # shutdown is a latch
        self.assertEqual(self.k.begin(6), 0)
        self.assertEqual(self.rig.names().count("open"), 1)

    def test_turning_off_closes_a_live_stream(self):
        self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.set_enabled(False)
        self.assertTrue(_wait_for(lambda: not self.rig.cell[0]))
        self.assertFalse(self.k.is_live())


class KeeperBoundsTests(unittest.TestCase):
    def tearDown(self):
        self.k.shutdown()

    def test_a_leaked_holder_counts_for_max_hold_only(self):
        rig = _Rig()
        self.k = rig.keeper(max_hold_s=0.2)
        self.k.set_enabled(True)
        self.k.begin(6)                       # never ended
        self.assertTrue(_wait_for(self.k.is_live))
        t0 = time.monotonic()
        self.assertTrue(_wait_for(lambda: not rig.cell[0], timeout=2.0))
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(rig.names()[-1], "release")

    def test_yield_closes_now_and_holds_off_until_reinit_done(self):
        rig = _Rig()
        self.k = rig.keeper(yield_s=5.0)
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.request_yield()
        self.assertTrue(_wait_for(lambda: not rig.cell[0]),
                        "the keeper must step aside at once")
        time.sleep(0.1)
        self.assertEqual(rig.names().count("open"), 1, "no reopen in yield")
        # The re-enumeration ran: the old index is stale, so nothing opens
        # until a caller hands over a fresh device ...
        self.k.reinit_done()
        time.sleep(0.1)
        self.assertEqual(rig.names().count("open"), 1)
        # ... and then it holds the speaker again, on that device.
        self.k.note_device(9)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertEqual([e[1] for e in rig.events if e[0] == "open"], [6, 9])
        self.k.end(tok)

    def test_yield_block_expires_by_itself(self):
        rig = _Rig()
        self.k = rig.keeper(yield_s=0.2)
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.request_yield()
        self.assertTrue(_wait_for(lambda: not rig.cell[0]))
        self.assertTrue(_wait_for(self.k.is_live, timeout=2.0))
        self.assertEqual(rig.names().count("open"), 2)
        self.k.end(tok)

    def test_refused_claim_backs_off_logs_once_and_never_opens(self):
        rig = _Rig(claim_ok=False)
        self.k = rig.keeper()
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("claim") >= 1))
        time.sleep(0.3)
        self.assertEqual(rig.names().count("claim"), 1, "no hot loop")
        self.assertNotIn("open", rig.names())
        self.assertFalse(rig.cell[0])
        self.assertEqual(sum("not holding the speaker" in ln
                             for ln in rig.lines), 1)
        self.k.end(tok)

    def test_failed_open_releases_the_cell_backs_off_and_recovers(self):
        rig = _Rig(open_error=RuntimeError("Error opening OutputStream"))
        clock = [100.0]
        self.k = rig.keeper(clock=lambda: clock[0])
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(lambda: "release" in rig.names()))
        self.assertEqual(rig.names(), ["claim", "open", "release"])
        self.assertFalse(rig.cell[0])
        time.sleep(0.2)
        self.assertEqual(rig.names().count("open"), 1, "no hot loop")
        self.assertEqual(sum("open failed on device 6" in ln
                             for ln in rig.lines), 1)
        # Past the first back-off (5 s on the fake clock) it tries again and,
        # once the device opens, says so.
        rig.open_error = None
        clock[0] += 5.1
        self.k.note_device(6)            # any notify re-evaluates
        self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertTrue(any("holding the speaker again" in ln
                            for ln in rig.lines), rig.lines)
        self.k.end(tok)

    def test_backoff_doubles(self):
        rig = _Rig(open_error=OSError("-9999"))
        clock = [0.0]
        self.k = rig.keeper(clock=lambda: clock[0])
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("open") == 1))
        clock[0] = 5.5
        self.k.note_device(6)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("open") == 2))
        clock[0] = 5.5 + 9.0              # inside the 10 s second back-off
        self.k.begin(6)
        time.sleep(0.15)
        self.assertEqual(rig.names().count("open"), 2)
        clock[0] = 5.5 + 10.5
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("open") == 3))


class KeeperCallerSafetyTests(unittest.TestCase):
    def tearDown(self):
        self.k.shutdown()

    def test_wait_settled_is_bounded_while_an_open_is_in_flight(self):
        rig = _Rig(open_delay=0.3)
        self.k = rig.keeper()
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: "open" in rig.names()))
        t0 = time.monotonic()
        self.assertFalse(self.k.wait_settled(0.05))
        self.assertLess(time.monotonic() - t0, 0.2)
        self.assertTrue(self.k.wait_settled(1.0))
        self.assertTrue(self.k.is_live())

    def test_a_wedged_open_stops_costing_playbacks_a_wait(self):
        rig = _Rig()
        gate = threading.Event()
        clock = [50.0]

        def _open(device):
            rig.event("open", device)
            gate.wait(5.0)
            return _FakeStream(rig, device)

        self.k = pk.PlaybackKeeper(open_stream=_open, claim=rig.claim,
                                   release=rig.release, log=rig.log,
                                   clock=lambda: clock[0], linger_s=0.05)
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: "open" in rig.names()))
        # In flight for less than OPEN_WEDGED_S: a bounded wait.
        t0 = time.monotonic()
        self.assertFalse(self.k.wait_settled(0.05))
        self.assertGreaterEqual(time.monotonic() - t0, 0.04)
        # Past it: no wait at all.
        clock[0] += pk.OPEN_WEDGED_S + 0.1
        t0 = time.monotonic()
        self.assertFalse(self.k.wait_settled(0.45))
        self.assertLess(time.monotonic() - t0, 0.02)
        gate.set()
        self.assertTrue(_wait_for(self.k.is_live))

    def test_a_wedged_close_keeps_the_cell_and_never_blocks_callers(self):
        rig = _Rig()
        rig.close_block = threading.Event()
        self.k = rig.keeper()
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)
        self.assertTrue(_wait_for(lambda: "close" in rig.names()))
        time.sleep(0.1)
        # The close never returned: the owner cell stays up (keeps
        # sd._terminate() away from the live native call) ...
        self.assertTrue(rig.cell[0])
        self.assertNotIn("release", rig.names())
        # ... and no caller blocks on it.
        t0 = time.monotonic()
        t2 = self.k.begin(6)
        self.k.end(t2)
        self.k.note_device(7)
        self.k.request_yield()
        self.k.wait_settled(0.01)
        self.assertLess(time.monotonic() - t0, 0.1)
        rig.close_block.set()
        self.assertTrue(_wait_for(lambda: not rig.cell[0]))

    def test_log_runs_off_the_lock(self):
        # A log hook that calls back into the keeper would deadlock if any
        # line were printed while the keeper's lock is held.
        rig = _Rig(open_error=RuntimeError("boom"))
        holder = {}
        finished = []

        def reentrant_log(line):
            rig.lines.append(line)
            k = holder["k"]

            def _calls():
                k.is_live()
                k.note_device(6)
                k.end(k.begin(6))
                finished.append(True)
            # On a helper thread with a bounded join: a deadlock fails the
            # test instead of hanging the suite.
            t = threading.Thread(target=_calls, daemon=True)
            t.start()
            t.join(1.0)

        self.k = pk.PlaybackKeeper(open_stream=rig.open_stream,
                                   claim=rig.claim, release=rig.release,
                                   log=reentrant_log, linger_s=0.05)
        holder["k"] = self.k
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: len(rig.lines) >= 1))
        self.assertTrue(_wait_for(lambda: finished == [True]),
                        "a keeper call from the log hook blocked: a line "
                        "was printed under the keeper's lock")

    def test_thread_start_failure_is_reported_not_raised(self):
        rig = _Rig()

        def factory(**kw):
            raise RuntimeError("can't start new thread")

        self.k = pk.PlaybackKeeper(open_stream=rig.open_stream,
                                   claim=rig.claim, release=rig.release,
                                   log=rig.log, thread_factory=factory)
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.k.end(tok)
        self.assertEqual(rig.events, [])
        self.assertTrue(any("could not start its thread" in ln
                            for ln in rig.lines), rig.lines)


class KeeperConstantsTests(unittest.TestCase):
    def test_per_turn_bounds(self):
        # Never permanent: a holder is bounded, the linger is short, and the
        # reaper poll it brings stays inside the 50 ms barge-in slice.
        self.assertLessEqual(pk.LINGER_S, 3.0)
        self.assertLessEqual(pk.MAX_HOLD_S, 120.0)
        self.assertGreater(pk.YIELD_S, 4.0)    # > DEVICE_CHECK_INTERVAL
        self.assertLessEqual(pk.REAP_POLL_S, 0.05)
        self.assertLess(pk.READY_WAIT_S, 1.0)


if __name__ == "__main__":
    unittest.main()
