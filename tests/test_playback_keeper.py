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
from unittest import mock

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
        t0 = time.monotonic()
        self.rig.event("abort", self.device)
        self.rig.span("abort", self.device, t0)

    def close(self, ignore_errors=True):
        t0 = time.monotonic()
        self.rig.event("close", self.device)
        if self.rig.close_hook is not None:
            self.rig.close_hook()
        if self.rig.close_delay:
            time.sleep(self.rig.close_delay)
        block = self.rig.close_block
        if block is not None:
            block.wait(5.0)
        self.rig.span("close", self.device, t0)


class _Rig:
    """Fakes for open_stream / claim / release / log, recording the order
    and the thread of every call."""

    def __init__(self, claim_ok=True, open_error=None, open_delay=0.0,
                 close_delay=0.0):
        self.events = []
        self.spans = []          # (name, device, t_start, t_end): native calls
        self.lock = threading.Lock()
        self.claim_ok = claim_ok
        self.open_error = open_error
        self.open_delay = open_delay
        self.close_delay = close_delay
        self.close_block = None
        self.close_hook = None
        self.claim_block = None
        self.cell = [False]
        self.lines = []

    def event(self, name, arg=None):
        with self.lock:
            self.events.append((name, arg, threading.get_ident()))

    def span(self, name, device, t0):
        with self.lock:
            self.spans.append((name, device, t0, time.monotonic()))

    def names(self):
        with self.lock:
            return [e[0] for e in self.events]

    def open_stream(self, device):
        t0 = time.monotonic()
        self.event("open", device)
        try:
            if self.open_delay:
                time.sleep(self.open_delay)
            if self.open_error is not None:
                raise self.open_error
            return _FakeStream(self, device)
        finally:
            self.span("open", device, t0)

    def claim(self):
        self.event("claim")
        if self.claim_block is not None:
            # A re-enumeration holds the owner gate's latch: the real claim
            # waits (bounded) for it to drop.
            self.claim_block.wait(5.0)
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
        tok7 = self.k.begin(7)            # the play path's begin(out_dev)
        self.assertFalse(self.k.is_live(), "not live on the device in use")
        self.assertTrue(_wait_for(
            lambda: [e[1] for e in self.rig.events if e[0] == "open"]
            == [6, 7]))
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)
        self.k.end(tok7)
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

    def test_turning_off_forgets_the_holders(self):
        # Off then on again (the owner flips the setting mid-reply): a
        # holder from before must not reopen the speaker by itself.
        self.k.begin(6)                       # never ended
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.set_enabled(False)
        self.assertTrue(_wait_for(lambda: not self.rig.cell[0]))
        self.k.set_enabled(True)
        time.sleep(0.15)                      # > linger_s
        self.assertEqual(self.rig.names().count("open"), 1)
        self.assertFalse(self.rig.cell[0])


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
        tok9 = self.k.begin(9)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertEqual([e[1] for e in rig.events if e[0] == "open"], [6, 9])
        self.k.end(tok)
        self.k.end(tok9)

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
        self.k.begin(6)                  # any notify re-evaluates
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
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("open") == 2))
        clock[0] = 5.5 + 9.0              # inside the 10 s second back-off
        self.k.begin(6)
        time.sleep(0.15)
        self.assertEqual(rig.names().count("open"), 2)
        clock[0] = 5.5 + 10.5
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("open") == 3))
        # Three failures in one streak: ONE line (review 2026-10-09: no test
        # pinned the count over repeated failures).
        self.assertTrue(_wait_for(lambda: rig.names().count("release") == 3))
        self.assertEqual(sum("not holding the speaker" in ln
                             for ln in rig.lines), 1, rig.lines)

    def test_each_failure_streak_logs_once(self):
        rig = _Rig(open_error=OSError("-9999"))
        clock = [0.0]
        self.k = rig.keeper(clock=lambda: clock[0])
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("release") == 1))
        clock[0] = 5.5                         # past the 5 s back-off
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("release") == 2))
        rig.open_error = None                  # it works again ...
        clock[0] = 5.5 + 10.5
        self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.set_enabled(False)              # ... closes ...
        self.assertTrue(_wait_for(lambda: not rig.cell[0]))
        rig.open_error = OSError("-9999")      # ... and a NEW streak starts
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: rig.names().count("open") == 4))
        self.assertTrue(_wait_for(lambda: rig.names().count("release") == 4))
        self.assertEqual(sum("not holding the speaker" in ln
                             for ln in rig.lines), 2, rig.lines)
        self.assertEqual(sum("holding the speaker again" in ln
                             for ln in rig.lines), 1, rig.lines)
        self.k.end(tok)


class KeeperCallerSafetyTests(unittest.TestCase):
    def tearDown(self):
        self.k.shutdown()

    def test_an_open_in_flight_is_waited_for_until_it_returns(self):
        # Review 2026-10-09: the budget is for a call only DUE. One already
        # in flight is waited for until it RETURNS - a caller that went
        # ahead after its budget would open alongside it.
        rig = _Rig(open_delay=0.3)
        self.k = rig.keeper()
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: "open" in rig.names()))
        t0 = time.monotonic()
        self.assertTrue(self.k.wait_settled(0.05))
        t1 = time.monotonic()
        self.assertGreater(t1 - t0, 0.15, "gave up on the open in flight")
        self.assertTrue(self.k.is_live())
        self.assertTrue(all(sp[3] <= t1 for sp in rig.spans), rig.spans)

    def test_the_wait_for_a_call_in_flight_is_bounded_in_real_time(self):
        # The keeper's clock stands still (a fake), so only the real-time
        # backstop (OPEN_WEDGED_S from the start of the wait) can end it.
        rig = _Rig()
        gate = threading.Event()
        self.addCleanup(gate.set)

        def _open(device):
            rig.event("open", device)
            gate.wait(5.0)
            return _FakeStream(rig, device)

        with mock.patch.object(pk, "OPEN_WEDGED_S", 0.3):
            self.k = pk.PlaybackKeeper(open_stream=_open, claim=rig.claim,
                                       release=rig.release, log=rig.log,
                                       clock=lambda: 50.0, linger_s=0.05)
            self.k.set_enabled(True)
            self.k.begin(6)
            self.assertTrue(_wait_for(lambda: "open" in rig.names()))
            t0 = time.monotonic()
            self.assertFalse(self.k.wait_settled(0.05))
            dt = time.monotonic() - t0
        self.assertGreaterEqual(dt, 0.25)
        self.assertLess(dt, 1.0)

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
        # In flight for less than OPEN_WEDGED_S: waited for (it returns).
        threading.Timer(0.1, gate.set).start()
        t0 = time.monotonic()
        self.assertTrue(self.k.wait_settled(0.05))
        self.assertGreaterEqual(time.monotonic() - t0, 0.08)
        self.assertTrue(self.k.is_live())
        # A second open, in flight past OPEN_WEDGED_S: no wait at all.
        gate.clear()
        self.k.end(self.k.begin(7))
        self.assertTrue(_wait_for(lambda: rig.names().count("open") == 2))
        clock[0] += pk.OPEN_WEDGED_S + 0.1
        t0 = time.monotonic()
        self.assertFalse(self.k.wait_settled(0.45))
        self.assertLess(time.monotonic() - t0, 0.02)
        gate.set()

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
        # ... and no caller that does not wait blocks on it.
        t0 = time.monotonic()
        t2 = self.k.begin(6)
        self.k.end(t2)
        self.k.end(self.k.begin(7))
        self.k.request_yield()
        self.k.is_live()
        self.assertLess(time.monotonic() - t0, 0.1)
        # The waiting ones (wait_settled / enter_play_open) wait for a close
        # in flight only until it counts as wedged: OPEN_WEDGED_S from its
        # start (0.3 s here), then not at all.
        with mock.patch.object(pk, "OPEN_WEDGED_S", 0.3):
            t0 = time.monotonic()
            self.assertFalse(self.k.wait_settled(0.01))
            self.assertLess(time.monotonic() - t0, 0.35)
            t0 = time.monotonic()
            self.assertFalse(self.k.wait_settled(0.01))
            self.assertLess(time.monotonic() - t0, 0.02)
        rig.close_block.set()
        self.assertTrue(_wait_for(lambda: not rig.cell[0]))

    def test_log_runs_off_the_lock(self):
        # A log hook that calls back into the keeper would deadlock if any
        # line were printed while the keeper's lock is held.
        rig = _Rig(open_error=RuntimeError("boom"))
        holder = {}
        in_time = []

        def reentrant_log(line):
            rig.lines.append(line)
            k = holder["k"]

            def _calls():
                k.is_live()
                k.note_play()
                k.end(k.begin(6))
            # On a helper thread with a SHORT bounded join, judged WHILE the
            # hook still runs: a line printed under the keeper's lock leaves
            # the helper blocked until the hook returns. (Review 2026-10-09:
            # the old version joined for 1 s and then waited for the helper
            # to finish at all - after the hook returned the lock dropped and
            # the helper finished late, so logging under the lock passed.)
            t = threading.Thread(target=_calls, daemon=True)
            t.start()
            t.join(0.5)
            in_time.append(not t.is_alive())

        self.k = pk.PlaybackKeeper(open_stream=rig.open_stream,
                                   claim=rig.claim, release=rig.release,
                                   log=reentrant_log, linger_s=0.05)
        holder["k"] = self.k
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: len(in_time) >= 1))
        self.assertTrue(all(in_time),
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


def _overlaps(spans, t0, t1):
    """Native keeper calls whose interval meets [t0, t1]."""
    return [s for s in spans if s[2] < t1 and s[3] > t0]


class KeeperOneOpenAtATimeTests(unittest.TestCase):
    """Review fix 2026-10-05 (ONE OPEN AT A TIME): a playback's own open and
    a keeper open or close never run at the same moment, in either order.
    Before the fix the playback waited only for a keeper open that had
    ALREADY started, so a device change (the keeper first closes, then
    opens) or a cold start ran straight into the playback's open."""

    rig = None

    def tearDown(self):
        self.k.shutdown()
        if self.rig is not None:
            if self.rig.close_block is not None:
                self.rig.close_block.set()
            if self.rig.claim_block is not None:
                self.rig.claim_block.set()

    def test_a_device_change_is_waited_for_before_a_play_opens(self):
        self.rig = rig = _Rig(open_delay=0.1, close_delay=0.05)
        self.k = rig.keeper()
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        # The play path: hold on the play's (new) device, then wait.
        tok2 = self.k.begin(7)
        t0 = time.monotonic()
        self.assertTrue(self.k.wait_settled(pk.READY_WAIT_S))
        self.assertLess(time.monotonic() - t0, pk.READY_WAIT_S + 0.2)
        # Nothing of the keeper's is in flight any more: the close of 6 and
        # the open of 7 both returned, and the speaker is held on 7.
        self.assertTrue(self.k.is_live(), "not held on the new device")
        self.assertEqual([e[1] for e in rig.events if e[0] == "open"], [6, 7])
        self.assertEqual([x[0] for x in rig.spans],
                         ["open", "abort", "close", "open"])
        self.k.end(tok)
        self.k.end(tok2)

    def test_a_cold_open_that_is_due_is_waited_for(self):
        self.rig = rig = _Rig(open_delay=0.1)
        self.k = rig.keeper()
        self.k.set_enabled(True)
        self.k.end(self.k.begin(6))           # thread up, stream closed again
        self.assertTrue(_wait_for(lambda: rig.names().count("release") == 1))
        tok = self.k.begin(6)
        tok_p, settled = self.k.enter_play_open(pk.READY_WAIT_S)
        t_play = time.monotonic()
        self.assertTrue(settled)
        self.assertTrue(self.k.is_live())
        time.sleep(0.05)                       # the play's own open
        self.assertEqual(_overlaps(rig.spans, t_play, time.monotonic()), [])
        self.k.exit_play_open(tok_p)
        self.k.end(tok)

    def test_the_keeper_makes_no_native_call_while_a_play_open_is_marked(self):
        self.rig = rig = _Rig()
        self.k = rig.keeper()
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        tok_p, settled = self.k.enter_play_open(pk.READY_WAIT_S)
        self.assertTrue(tok_p)
        self.assertTrue(settled)
        t_mark = time.monotonic()
        # A device change (or a block that runs out) lands while the
        # playback's open is in flight: the keeper must not touch PortAudio.
        tok7 = self.k.begin(7)
        time.sleep(0.15)
        t_exit = time.monotonic()
        self.assertEqual(_overlaps(rig.spans, t_mark, t_exit), [])
        self.assertEqual(rig.names().count("abort"), 0)
        self.k.exit_play_open(tok_p)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertEqual([e[1] for e in rig.events if e[0] == "open"], [6, 7])
        self.assertTrue(all(x[2] >= t_exit for x in rig.spans[1:]), rig.spans)
        self.k.end(tok)
        self.k.end(tok7)

    def test_a_wedged_play_open_stops_holding_the_keeper_back(self):
        self.rig = rig = _Rig()
        clock = [10.0]
        self.k = rig.keeper(clock=lambda: clock[0])
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        tok_p, _ = self.k.enter_play_open(0.0)     # never exited (wedged)
        tok7 = self.k.begin(7)
        time.sleep(0.1)
        self.assertEqual(rig.names().count("abort"), 0)
        clock[0] += pk.OPEN_WEDGED_S + 0.1
        self.k.set_enabled(True)                   # any notify re-evaluates
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertEqual([e[1] for e in rig.events if e[0] == "open"], [6, 7])
        self.k.exit_play_open(tok_p)
        self.k.end(tok)
        self.k.end(tok7)

    def test_a_keeper_close_in_flight_is_waited_for_by_a_play(self):
        self.rig = rig = _Rig()
        rig.close_block = threading.Event()
        clock = [10.0]
        self.k = rig.keeper(clock=lambda: clock[0])
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)
        clock[0] += 1.0                            # past the 0.05 s linger
        self.k.set_enabled(True)
        self.assertTrue(_wait_for(lambda: "close" in rig.names()))
        # In flight for less than OPEN_WEDGED_S: waited for past the play's
        # own 0.05 s budget, until it returns.
        threading.Timer(0.1, rig.close_block.set).start()
        t0 = time.monotonic()
        tok_p, settled = self.k.enter_play_open(0.05)
        t1 = time.monotonic()
        self.assertTrue(settled)
        self.assertGreaterEqual(t1 - t0, 0.08)
        self.assertTrue(all(sp[3] <= t1 for sp in rig.spans), rig.spans)
        self.k.exit_play_open(tok_p)

    def test_a_wedged_keeper_close_costs_plays_no_wait(self):
        self.rig = rig = _Rig()
        rig.close_block = threading.Event()
        clock = [10.0]
        self.k = rig.keeper(clock=lambda: clock[0])
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)
        clock[0] += 1.0                            # past the 0.05 s linger
        self.k.set_enabled(True)
        self.assertTrue(_wait_for(lambda: "close" in rig.names()))
        # In flight past OPEN_WEDGED_S: no wait at all.
        clock[0] += pk.OPEN_WEDGED_S + 0.1
        t0 = time.monotonic()
        tok_p, settled = self.k.enter_play_open(pk.READY_WAIT_S)
        self.assertFalse(settled)
        self.assertLess(time.monotonic() - t0, 0.05)
        self.k.exit_play_open(tok_p)
        rig.close_block.set()
        self.assertTrue(_wait_for(lambda: not rig.cell[0]))

    def test_a_play_never_opens_alongside_a_slow_keeper_open(self):
        # Review 2026-10-09 (both reviewers): the old flat READY_WAIT_S
        # budget gave up on a keeper open still in flight (live: 1 in 168
        # first-line opens took ~0.7 s) and let the play open alongside it.
        self.rig = rig = _Rig(open_delay=0.6)
        self.k = rig.keeper()
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(lambda: "open" in rig.names()))
        t0 = time.monotonic()
        tok_p, settled = self.k.enter_play_open(pk.READY_WAIT_S)
        t_play = time.monotonic()
        self.assertTrue(settled)
        self.assertGreater(t_play - t0, pk.READY_WAIT_S - 0.1)
        self.assertTrue(self.k.is_live())
        time.sleep(0.05)                           # the play's own open
        t_play_end = time.monotonic()
        self.k.exit_play_open(tok_p)
        # Judged once every keeper call has returned (spans are recorded at
        # the END of a call, so a call still running would not show yet).
        self.assertTrue(_wait_for(lambda: any(sp[0] == "open"
                                              for sp in rig.spans)))
        self.assertEqual(_overlaps(rig.spans, t_play, t_play_end), [])
        self.k.end(tok)

    def test_a_call_only_due_is_waited_for_its_budget_then_held_off(self):
        # Giving up on a DUE call (not started) is safe: the play's mark
        # holds it off until exit_play_open.
        self.rig = rig = _Rig()
        self.k = rig.keeper()
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        tok_a, _ = self.k.enter_play_open(pk.READY_WAIT_S)   # play A opening
        tok7 = self.k.begin(7)          # speaker change: close + open DUE
        t0 = time.monotonic()
        tok_b, settled = self.k.enter_play_open(0.1)         # play B
        dt = time.monotonic() - t0
        self.assertFalse(settled)
        self.assertGreaterEqual(dt, 0.09)
        self.assertLess(dt, 0.5)
        self.k.exit_play_open(tok_a)
        time.sleep(0.1)
        self.assertEqual(rig.names().count("abort"), 0,
                         "the keeper moved while play B's open was marked")
        self.k.exit_play_open(tok_b)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertEqual([e[1] for e in rig.events if e[0] == "open"], [6, 7])
        self.k.end(tok)
        self.k.end(tok7)

    def test_no_keeper_thread_means_nothing_to_wait_for_or_mark(self):
        self.rig = rig = _Rig()
        self.k = rig.keeper()                      # never enabled
        self.assertEqual(self.k.enter_play_open(1.0), (0, True))
        self.k.exit_play_open(0)
        self.assertEqual(rig.events, [])


class KeeperStaleIndexTests(unittest.TestCase):
    def tearDown(self):
        self.k.shutdown()
        if self.rig.claim_block is not None:
            self.rig.claim_block.set()

    def test_an_open_claimed_after_a_reinit_opens_nothing(self):
        # Review fix 2026-10-05: the keeper decided to open device 6, then its
        # claim waited out a re-enumeration's latch. reinit_done() ran while
        # the latch was held, so index 6 may now name another device: the
        # keeper must open nothing until it is handed a fresh index.
        self.rig = rig = _Rig()
        rig.claim_block = threading.Event()
        self.k = rig.keeper()
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(lambda: "claim" in rig.names()))
        self.k.reinit_done()                       # under the latch
        rig.claim_block.set()                      # latch drops: claim ok
        self.assertTrue(_wait_for(lambda: "release" in rig.names()))
        time.sleep(0.05)
        self.assertNotIn("open", rig.names())
        self.assertFalse(rig.cell[0])
        self.assertEqual(self.k.dropped_opens, 1)
        self.assertEqual(self.k.failures, 0, "a dropped open is no failure")
        tok9 = self.k.begin(9)                     # a fresh index
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertEqual([e[1] for e in rig.events if e[0] == "open"], [9])
        self.k.end(tok)
        self.k.end(tok9)


class KeeperLingerTests(unittest.TestCase):
    def tearDown(self):
        self.k.shutdown()

    def test_the_stream_lingers_after_the_last_holder_then_closes(self):
        rig = _Rig()
        clock = [100.0]
        self.k = rig.keeper(clock=lambda: clock[0], linger_s=2.0)
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)                            # linger until 102.0
        time.sleep(0.1)
        self.assertTrue(self.k.is_live(), "closed with no linger")
        clock[0] = 101.9
        self.k.set_enabled(True)                   # notify: re-evaluate
        time.sleep(0.1)
        self.assertTrue(self.k.is_live(), "closed before the linger ran out")
        self.assertNotIn("close", rig.names())
        clock[0] = 102.1
        self.k.set_enabled(True)
        self.assertTrue(_wait_for(lambda: not rig.cell[0]))
        self.assertEqual(rig.names().count("close"), 1)


class KeeperShutdownWaitTests(unittest.TestCase):
    rig = None

    def tearDown(self):
        self.k.shutdown()
        if self.rig is not None and self.rig.close_block is not None:
            self.rig.close_block.set()

    def test_shutdown_can_wait_for_the_close(self):
        # The interpreter-exit path: the keeper's close must have returned
        # before sounddevice's own exit handler terminates PortAudio.
        self.rig = rig = _Rig(close_delay=0.1)
        self.k = rig.keeper()
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertTrue(self.k.shutdown(wait_s=2.0))
        self.assertFalse(rig.cell[0])             # no polling: already done
        self.assertEqual(rig.names()[-1], "release")

    def test_the_shutdown_wait_is_bounded(self):
        self.rig = rig = _Rig()
        rig.close_block = threading.Event()
        self.k = rig.keeper()
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        t0 = time.monotonic()
        self.assertFalse(self.k.shutdown(wait_s=0.1))
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertTrue(rig.cell[0], "a wedged close keeps the cell up")

    def test_shutdown_of_a_keeper_that_never_ran_is_immediate(self):
        self.rig = _Rig()
        self.k = self.rig.keeper()
        self.assertTrue(self.k.shutdown(wait_s=1.0))


class KeeperYieldWaitTests(unittest.TestCase):
    """Review 2026-10-09 (audio safety, medium): _refresh_devices asks the
    keeper to step aside and then waits - wait_settled(YIELD_WAIT_S) - for
    its close to RETURN, because its caller (record_speech) opens the mic
    straight after. The wait the monolith makes is pinned in
    tests/monolith/test_monolith_playback_keeper.py (KeeperYieldWaitTests);
    this pins that wait_settled really covers the close."""

    def tearDown(self):
        self.k.shutdown()

    def test_the_wait_returns_only_after_the_close_returned(self):
        rig = _Rig(close_delay=0.08)
        self.k = rig.keeper(linger_s=2.0)
        self.k.set_enabled(True)
        self.k.end(self.k.begin(6))           # lingering: only the keeper
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.request_yield()
        self.assertTrue(self.k.wait_settled(pk.YIELD_WAIT_S))
        t_after = time.monotonic()
        closes = [sp for sp in rig.spans if sp[0] == "close"]
        self.assertEqual(len(closes), 1, rig.spans)
        self.assertLessEqual(closes[0][3], t_after)
        self.assertFalse(rig.cell[0])
        self.assertEqual(rig.names()[-1], "release")

    def test_a_yield_during_the_keepers_open_waits_for_open_and_close(self):
        rig = _Rig(open_delay=0.25, close_delay=0.02)
        self.k = rig.keeper(linger_s=2.0)
        self.k.set_enabled(True)
        self.k.end(self.k.begin(6))
        self.assertTrue(_wait_for(lambda: "open" in rig.names()))
        self.k.request_yield()                # the claim is up: deferred
        self.assertTrue(self.k.wait_settled(pk.YIELD_WAIT_S))
        t_after = time.monotonic()
        self.assertEqual([sp[0] for sp in rig.spans],
                         ["open", "abort", "close"])
        self.assertTrue(all(sp[3] <= t_after for sp in rig.spans), rig.spans)
        self.assertFalse(rig.cell[0])


class KeeperQuarantineTests(unittest.TestCase):
    """Review 2026-10-09: a keeper native call that ran OPEN_WEDGED_S or
    longer is one a playback could not wait for (it opened alongside it), so
    the keeper switches itself off for the rest of the process: it closes
    what it holds and never opens again. Logged once."""

    def tearDown(self):
        self.k.shutdown()

    def _keeper(self, rig, clock, open_s=0.0):
        def _open(device):
            rig.event("open", device)
            clock[0] += open_s                   # the native open's length
            if rig.open_error is not None:
                raise rig.open_error
            return _FakeStream(rig, device)

        return pk.PlaybackKeeper(open_stream=_open, claim=rig.claim,
                                 release=rig.release, log=rig.log,
                                 clock=lambda: clock[0], linger_s=0.05)

    def test_a_wedged_open_switches_the_keeper_off_for_good(self):
        rig = _Rig()
        clock = [10.0]
        # The close of what the slow open gave it is slow too: still ONE
        # quarantine line.
        rig.close_hook = lambda: clock.__setitem__(
            0, clock[0] + pk.OPEN_WEDGED_S + 0.1)
        self.k = self._keeper(rig, clock, open_s=pk.OPEN_WEDGED_S + 0.1)
        self.k.set_enabled(True)
        self.k.begin(6)
        # It closes what the slow open gave it, and drops the owner cell.
        self.assertTrue(_wait_for(lambda: "release" in rig.names()))
        self.assertEqual(rig.names(),
                         ["claim", "open", "abort", "close", "release"])
        self.assertTrue(self.k.quarantined)
        self.assertFalse(rig.cell[0])
        # Nothing opens again, whatever asks.
        self.k.set_enabled(True)
        self.assertEqual(self.k.begin(6), 0)
        self.assertEqual(self.k.begin(7), 0)
        time.sleep(0.1)
        self.assertEqual(rig.names().count("open"), 1)
        self.assertEqual(self.k.enter_play_open(0.1), (0, True))
        self.assertEqual(sum("OFF for the rest of this run" in ln
                             for ln in rig.lines), 1, rig.lines)

    def test_a_wedged_failed_open_quarantines_too(self):
        rig = _Rig(open_error=OSError("-9999"))
        clock = [10.0]
        self.k = self._keeper(rig, clock, open_s=pk.OPEN_WEDGED_S + 0.5)
        self.k.set_enabled(True)
        self.k.begin(6)
        self.assertTrue(_wait_for(lambda: "release" in rig.names()))
        self.assertTrue(_wait_for(lambda: self.k.quarantined))
        clock[0] += 120.0                        # far past any back-off
        self.assertEqual(self.k.begin(6), 0)
        time.sleep(0.1)
        self.assertEqual(rig.names().count("open"), 1)

    def test_a_wedged_close_switches_the_keeper_off_for_good(self):
        rig = _Rig()
        clock = [10.0]
        rig.close_hook = lambda: clock.__setitem__(
            0, clock[0] + pk.OPEN_WEDGED_S + 0.1)
        self.k = self._keeper(rig, clock)
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.k.end(tok)
        clock[0] += 1.0                          # past the linger
        self.k.set_enabled(True)                 # notify: re-evaluate
        self.assertTrue(_wait_for(lambda: "release" in rig.names()))
        self.assertTrue(_wait_for(lambda: self.k.quarantined))
        rig.close_hook = None
        self.assertEqual(self.k.begin(6), 0)
        time.sleep(0.1)
        self.assertEqual(rig.names().count("open"), 1)
        self.assertEqual(sum("its close on device 6" in ln
                             for ln in rig.lines), 1, rig.lines)

    def test_a_slow_open_under_the_wedge_bound_is_kept(self):
        rig = _Rig()
        clock = [10.0]
        self.k = self._keeper(rig, clock, open_s=pk.OPEN_WEDGED_S - 0.2)
        self.k.set_enabled(True)
        tok = self.k.begin(6)
        self.assertTrue(_wait_for(self.k.is_live))
        self.assertFalse(self.k.quarantined)
        self.assertNotIn("close", rig.names())
        self.k.end(tok)


class KeeperConstantsTests(unittest.TestCase):
    def test_per_turn_bounds(self):
        # Never permanent: a holder is bounded, the linger is short, and the
        # reaper poll it brings stays inside the 50 ms barge-in slice.
        self.assertLessEqual(pk.LINGER_S, 3.0)
        # ...but there IS a linger: it covers the gap between a reply and its
        # follow-up round, or a filler and the answer.
        self.assertGreaterEqual(pk.LINGER_S, 1.0)
        self.assertLessEqual(pk.MAX_HOLD_S, 120.0)
        self.assertGreater(pk.YIELD_S, 4.0)    # > DEVICE_CHECK_INTERVAL
        self.assertLessEqual(pk.REAP_POLL_S, 0.05)
        # A playback waits for a keeper open (measured 322 ms) - and not
        # much longer than one, or a slow keeper costs more than it saves.
        self.assertGreaterEqual(pk.READY_WAIT_S, 0.35)
        self.assertLessEqual(pk.READY_WAIT_S, 0.5)
        self.assertGreater(pk.OPEN_WEDGED_S, pk.READY_WAIT_S)
        # The re-enumeration's wait for the keeper's close (10 ms) also
        # covers a keeper open in flight when the yield came (live max
        # 699 ms) - and stays a short, bounded wait on the voice thread.
        self.assertGreaterEqual(pk.YIELD_WAIT_S, 0.8)
        self.assertLessEqual(pk.YIELD_WAIT_S, pk.OPEN_WEDGED_S)


if __name__ == "__main__":
    unittest.main()
