"""Tests for tools/test_watchdog.py — the TIME ceiling on a test run.

WHY THIS FILE EXISTS
====================
2026-09-04: ``python tools/run_tests_ci_sim.py`` was run with a 900-second outer
timeout and came back EXIT 124. All three CI gates had printed OK, so the stall
was inside the test run — and the run said NOTHING about where. unittest prints
its dots as it goes but a hang produces no summary, no traceback and no test id,
so an exit-124 means "somewhere in 15,000 tests" and nothing more.

The runners were bounded in exactly one dimension: tools/mem_guard.py, a MEMORY
ceiling. A hang is not an allocation, so it could never fire. tools/
test_watchdog.py is the missing dimension, and this file is what keeps it
honest — including the structural half, because a watchdog quietly dropped from
a runner is a watchdog that will not be there the next time it is needed (the
same shape as tests/test_mem_guard.py's runner scan, deliberately).

Everything here is deterministic: the fire path is exercised through an injected
``_exit`` (or, below the exit, a fake kernel32 plus an injected ``_os_exit``)
and a zeroed dump grace, so no test in this file can ever kill the process
running it, and no test here sleeps for a real budget. The two real-process
tests kill a CHILD interpreter, never this one.
"""
from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tools import test_watchdog as tw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "tools")

# Kept identical to tests/test_mem_guard.py's scan so the two cannot disagree
# about what counts as a runner.
_DISCOVERY_MARKERS = ("unittest.TestLoader", "unittest.main(")
_GUARDED_RUNNERS = ("run_tests.py", "run_tests_ci_sim.py", "run_coverage.py")


def _source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class _WatchdogBase(unittest.TestCase):
    """Detach from the LIVE watchdog for the duration of a test, then hand the
    module back exactly as it was.

    This suite normally runs under a runner that has already armed the real
    watchdog. Calling disarm() here would stop it for the rest of the run, and
    poking the live cell would tell it to kill the very run it is guarding — so
    these tests swap in a private, unarmed state and restore the original on the
    way out, stopping only a watchdog they armed themselves."""

    def setUp(self):
        self._cell = list(tw._cell)
        self._state = dict(tw._state)
        tw._state.update({"armed": False, "thread": None, "stop": None,
                          "per_phase": 0.0, "total": 0.0, "run_start": 0.0})
        self.addCleanup(self._restore)

    def _restore(self):
        stop = tw._state.get("stop")
        if stop is not None and stop is not self._state.get("stop"):
            stop.set()                      # only ever our own
        tw._cell[:] = self._cell
        tw._state.clear()
        tw._state.update(self._state)


# ── budget parsing ──────────────────────────────────────────────────────────
class BudgetTests(_WatchdogBase):
    def _budget(self, raw, default=99.0):
        env = {} if raw is None else {"X": raw}
        with mock.patch.dict(os.environ, env, clear=False):
            if raw is None:
                os.environ.pop("X", None)
            return tw._budget("X", default)

    def test_unset_uses_the_default(self):
        self.assertEqual(self._budget(None), 99.0)

    def test_blank_uses_the_default(self):
        self.assertEqual(self._budget("   "), 99.0)

    def test_a_number_wins(self):
        self.assertEqual(self._budget("12.5"), 12.5)

    def test_every_documented_off_spelling_disables(self):
        for word in ("0", "off", "OFF", "none", "unlimited", "no", "false"):
            with self.subTest(word=word):
                self.assertEqual(self._budget(word), 0.0)

    def test_garbage_falls_back_to_the_default_not_to_unbounded(self):
        # Fail CLOSED: the failure mode of a typo must be "still protected".
        for junk in ("banana", "12s", "--", "1e", "NaNish"):
            with self.subTest(junk=junk):
                self.assertEqual(self._budget(junk), 99.0)

    def test_a_negative_falls_back_to_the_default(self):
        self.assertEqual(self._budget("-5"), 99.0)

    def test_the_shipped_defaults_are_the_measured_ones(self):
        # 2026-09-04 measurement: 15,361 tests in 136.4s, slowest single test
        # 5.00s. The per-phase default must be far above the slowest test and
        # the total default far above the whole run, but BELOW ci.yml's
        # 20-minute job timeout so the watchdog always reports first.
        self.assertGreaterEqual(tw.DEFAULT_PER_PHASE_S, 60.0)
        self.assertGreaterEqual(tw.DEFAULT_TOTAL_S, 600.0)
        # Read the job timeout from ci.yml itself (it was raised 20 -> 40 on
        # 2026-10-14 when the two-run job outgrew 20 minutes).
        with open(os.path.join(_PROJECT_ROOT, ".github", "workflows", "ci.yml"),
                  encoding="utf-8") as f:
            minutes = [int(line.split(":", 1)[1]) for line in f
                       if line.strip().startswith("timeout-minutes:")]
        self.assertEqual(len(minutes), 1, minutes)
        self.assertLess(tw.DEFAULT_TOTAL_S, minutes[0] * 60,
                        "the whole-run budget must fire before ci.yml's "
                        "timeout-minutes kills the job with no diagnosis")
        self.assertGreaterEqual(minutes[0], 30,
                                "the job runs the suite twice (unit + coverage); "
                                "20 minutes stopped being enough on 2026-10-14")


# ── the trigger decision ────────────────────────────────────────────────────
class ExpiryDecisionTests(unittest.TestCase):
    """_expired is pure, so these never touch the module's live cell — see
    _WatchdogBase for why that matters."""

    @staticmethod
    def _call(now, *, label="test x", started=100.0, per_phase=10.0,
              total=100.0, run_start=100.0):
        return tw._expired(now, label=label, started=started,
                           per_phase=per_phase, total=total,
                           run_start=run_start)

    def test_inside_both_budgets_is_none(self):
        self.assertIsNone(self._call(105.0))

    def test_a_phase_over_budget_fires_and_names_the_phase(self):
        reason = self._call(111.0, label="test tests.foo.BarTests.test_baz")
        self.assertIsNotNone(reason)
        self.assertIn("tests.foo.BarTests.test_baz", reason)
        self.assertIn("phase exceeded", reason)

    def test_the_whole_run_budget_fires_even_when_no_phase_does(self):
        # Death by a thousand slow tests: every phase is short, the run is not.
        reason = self._call(151.0, per_phase=1000.0, total=50.0,
                            started=140.0, run_start=100.0)
        self.assertIsNotNone(reason)
        self.assertIn("whole run exceeded", reason)

    def test_a_disabled_per_phase_budget_never_fires_on_a_phase(self):
        self.assertIsNone(self._call(1_000_000.0, per_phase=0.0, total=0.0))

    def test_an_unstarted_phase_cannot_fire(self):
        self.assertIsNone(self._call(1_000_000.0, started=0.0, total=0.0))


# ── the fire path ───────────────────────────────────────────────────────────
class FireTests(_WatchdogBase):
    def test_fire_dumps_twice_and_exits_with_the_documented_code(self):
        codes: list = []
        with mock.patch.object(tw, "_exit", codes.append), \
             mock.patch.object(tw, "DUMP_GRACE_S", 0.0), \
             mock.patch.object(tw, "_dump") as dump, \
             mock.patch.object(tw, "_write") as write:
            self.assertTrue(tw._fire("phase exceeded 1s: test tests.foo.test_bar"))
        self.assertEqual(codes, [tw.EXIT_CODE])
        self.assertEqual(dump.call_count, 2,
                         "two dumps: identical stacks mean WEDGED, different "
                         "ones mean pathologically SLOW")
        said = " ".join(str(c.args[1]) for c in write.call_args_list
                        if len(c.args) > 1)
        self.assertIn("tests.foo.test_bar", said,
                      "the banner must name the phase — naming it is the whole "
                      "reason this exists")

    def test_a_run_that_frees_itself_in_the_grace_window_is_not_killed(self):
        """SLOW is not WEDGED. A test that merely overshoots a tight budget and
        then finishes must NOT be turned into a red run — measured live on
        2026-09-04: with JARVIS_TEST_TIMEOUT_S=2, a real 3.5s test tripped the
        banner and then completed, and the process must survive that."""
        codes: list = []
        with mock.patch.object(tw, "_exit", codes.append), \
             mock.patch.object(tw, "DUMP_GRACE_S", 0.0), \
             mock.patch.object(tw, "_dump") as dump, \
             mock.patch.object(tw, "_write") as write:
            self.assertFalse(tw._fire("phase exceeded 1s: test x",
                                      lambda: False))
        self.assertEqual(codes, [], "it killed a run that had recovered")
        self.assertEqual(dump.call_count, 1,
                         "the first dump is still worth having; the second is "
                         "the confirmation that never came")
        said = " ".join(str(c.args[1]) for c in write.call_args_list
                        if len(c.args) > 1)
        self.assertIn("SLOW, not wedged", said)

    def test_an_unanswerable_recheck_fails_closed_and_kills(self):
        codes: list = []
        with mock.patch.object(tw, "_exit", codes.append), \
             mock.patch.object(tw, "DUMP_GRACE_S", 0.0), \
             mock.patch.object(tw, "_dump"), mock.patch.object(tw, "_write"):
            def _boom():
                raise RuntimeError("the state cell is unreadable")
            self.assertTrue(tw._fire("phase exceeded 1s: test x", _boom))
        self.assertEqual(codes, [tw.EXIT_CODE],
                         "a recheck that cannot answer must not be read as 'all "
                         "clear' — the run is already known to be over budget")

    def test_the_exit_code_is_distinguishable_from_a_test_failure(self):
        # 1 = unittest failures, 124 = a shell `timeout`. 3 must be neither, so
        # a log tells you which bound tripped.
        self.assertNotIn(tw.EXIT_CODE, (0, 1, 124))

    def test_stderr_fd_survives_a_mocked_stream(self):
        broken = mock.Mock()
        broken.fileno.side_effect = ValueError("I/O operation on closed file")
        with mock.patch("sys.stderr", broken), \
             mock.patch("sys.__stderr__", broken):
            self.assertEqual(tw._stderr_fd(), 2)

    def test_write_and_dump_never_raise(self):
        # Both run on the failure path; an exception there would swallow the
        # only diagnosis the run is ever going to produce.
        tw._write(-1, "this fd does not exist")
        with mock.patch.object(tw.faulthandler, "dump_traceback",
                               side_effect=RuntimeError("boom")):
            tw._dump(2)


# ── the kill itself: TerminateProcess, os._exit only as the fallback ─────────
class _Landed(BaseException):
    """What a real TerminateProcess(GetCurrentProcess()) does: never return.
    BaseException so _hard_exit's ``except Exception`` cannot swallow it."""


class _FakeKernel32:
    """The seam: the kernel32 object _bind_kernel32 hands the module. One step
    above the kernel, so a test sees exactly the call that would kill."""

    PSEUDO_HANDLE = -1          # what GetCurrentProcess() really returns

    def __init__(self, *, lands=True, raises=None):
        self.lands = lands
        self.raises = raises
        self.calls: list = []

    def GetCurrentProcess(self):
        self.calls.append(("GetCurrentProcess",))
        return self.PSEUDO_HANDLE

    def TerminateProcess(self, handle, code):
        self.calls.append(("TerminateProcess", handle, code))
        if self.raises is not None:
            raise self.raises
        if self.lands:
            raise _Landed(code)
        return 0                # FALSE: the kernel refused the kill


class HardExitTests(_WatchdogBase):
    """2026-10-13: the v2.0.191 gate printed "exiting 3" at 09:43:39 and its
    python.exe was still alive ~26 h later — os._exit is ExitProcess, which
    can wedge in DLL detach under the loader lock. These pin the replacement:
    TerminateProcess on our own pseudo-handle, os._exit only as the fallback."""

    def _hard_exit(self, k32):
        fallback: list = []
        with mock.patch.object(tw, "_k32", k32), \
             mock.patch.object(tw, "_os_exit", fallback.append):
            try:
                tw._hard_exit(tw.EXIT_CODE)
                landed = False
            except _Landed:
                landed = True
        return landed, fallback

    def test_the_watchdogs_exit_is_the_hard_exit_not_a_raw_os_exit(self):
        self.assertIs(tw._exit, tw._hard_exit,
                      "the fire path must end in _hard_exit — a bare os._exit "
                      "here is the 26-hour zombie of 2026-10-13")

    def test_it_terminates_the_current_process_with_the_exit_code(self):
        k32 = _FakeKernel32()
        landed, fallback = self._hard_exit(k32)
        self.assertTrue(landed, "TerminateProcess was never called")
        self.assertEqual(k32.calls, [
            ("GetCurrentProcess",),
            ("TerminateProcess", _FakeKernel32.PSEUDO_HANDLE, tw.EXIT_CODE)])
        self.assertEqual(fallback, [],
                         "os._exit ran although TerminateProcess had landed")

    def test_a_refused_kill_falls_back_to_os_exit(self):
        k32 = _FakeKernel32(lands=False)
        landed, fallback = self._hard_exit(k32)
        self.assertFalse(landed)
        self.assertEqual(k32.calls[-1][0], "TerminateProcess",
                         "the fallback must come AFTER the hard kill was tried")
        self.assertEqual(fallback, [tw.EXIT_CODE])

    def test_a_raising_kernel32_still_falls_back_to_os_exit(self):
        k32 = _FakeKernel32(raises=OSError("access denied"))
        landed, fallback = self._hard_exit(k32)
        self.assertFalse(landed)
        self.assertEqual(fallback, [tw.EXIT_CODE],
                         "an exception on the kill path must never leave the "
                         "process alive")

    def test_no_kernel32_means_os_exit(self):
        landed, fallback = self._hard_exit(None)
        self.assertFalse(landed)
        self.assertEqual(fallback, [tw.EXIT_CODE])

    def test_a_blown_budget_reaches_terminate_process(self):
        """The whole fire path, NOT patching _exit: dump, dump, then the kill
        at the kernel32 seam. This is the call the 2026-10-13 run never made."""
        k32 = _FakeKernel32()
        fallback: list = []
        with mock.patch.object(tw, "_k32", k32), \
             mock.patch.object(tw, "_os_exit", fallback.append), \
             mock.patch.object(tw, "DUMP_GRACE_S", 0.0), \
             mock.patch.object(tw, "_dump"), \
             mock.patch.object(tw, "_write") as write:
            with self.assertRaises(_Landed):
                tw._fire("phase exceeded 1s: test tests.foo.test_bar",
                         lambda: True)
        self.assertIn(("TerminateProcess", _FakeKernel32.PSEUDO_HANDLE,
                       tw.EXIT_CODE), k32.calls)
        self.assertEqual(fallback, [])
        said = " ".join(str(c.args[1]) for c in write.call_args_list
                        if len(c.args) > 1)
        self.assertIn("exiting %d via TerminateProcess" % tw.EXIT_CODE, said,
                      "the log must say which kill was attempted")

    @unittest.skipUnless(os.name == "nt", "TerminateProcess is Windows-only")
    def test_kernel32_is_bound_with_pointer_sized_handles(self):
        # os.name, not sys.platform: ci-sim fakes the latter after this module
        # bound kernel32, which is exactly why the binding happens at import.
        from ctypes import wintypes
        self.assertIsNotNone(tw._k32, "no hard kill available on Windows")
        self.assertIs(tw._k32.GetCurrentProcess.restype, wintypes.HANDLE)
        self.assertEqual(list(tw._k32.TerminateProcess.argtypes),
                         [wintypes.HANDLE, wintypes.UINT])

    def test_the_only_os_exit_left_is_the_fallback_binding(self):
        """Structural: one ``os._exit`` reference in the module, and it is the
        ``_os_exit = os._exit`` binding _hard_exit falls back to."""
        tree = ast.parse(_source(os.path.join(_TOOLS_DIR, "test_watchdog.py")))
        uses = [n for n in ast.walk(tree)
                if isinstance(n, ast.Attribute) and n.attr == "_exit"
                and isinstance(n.value, ast.Name) and n.value.id == "os"]
        self.assertEqual(len(uses), 1, "a raw os._exit crept back in")
        binding = [n for n in tree.body if isinstance(n, ast.Assign)
                   and n.value is uses[0]]
        self.assertEqual([t.id for t in binding[0].targets] if binding else [],
                         ["_os_exit"])


class RealProcessExitTests(unittest.TestCase):
    """End to end in a CHILD interpreter: arm with a 0.2 s budget, wedge the
    main thread, and require the process to be gone with EXIT_CODE.

    Bounded the 11e way (memory: subprocess.run's timeout is not a bound on
    Windows): output to a file, never a pipe, then Popen.wait(timeout)."""

    _CHILD = "\n".join((
        "import os, sys, threading",
        "sys.path.insert(0, sys.argv[1])",
        "wedge = threading.Event()",
        "if sys.argv[2] == 'wedged-os-exit':",
        "    def _never_returns(code):",
        "        os.write(1, b'os._exit reached\\n')",
        "        wedge.wait()",
        "    os._exit = _never_returns",
        "from tools import test_watchdog as tw",
        "tw.DUMP_GRACE_S = 0.0",
        "tw.arm(per_phase_s=0.2, total_s=0.0)",
        "wedge.wait()",
    ))

    def _run_child(self, mode: str):
        env = {k: v for k, v in os.environ.items()
               if k not in (tw.PER_PHASE_ENV, tw.TOTAL_ENV)}
        with tempfile.TemporaryFile() as out:
            proc = subprocess.Popen(
                [sys.executable, "-c", self._CHILD, _PROJECT_ROOT, mode],
                cwd=_PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                rc = None
            out.seek(0)
            text = out.read().decode("utf-8", "replace")
        return rc, text

    def test_a_wedged_run_really_dies_with_the_exit_code(self):
        rc, text = self._run_child("plain")
        self.assertEqual(rc, tw.EXIT_CODE,
                         "the watchdog fired but the process did not end with "
                         "EXIT_CODE (None = still alive after 60 s):\n" + text[-2000:])

    @unittest.skipUnless(os.name == "nt", "the ExitProcess wedge is Windows-only")
    def test_it_dies_even_when_os_exit_would_never_return(self):
        """The 2026-10-13 shape: os._exit hangs (here a stand-in that never
        returns, installed before the import). The old watchdog printed
        'exiting 3', reached it and wedged; this one must not reach it."""
        rc, text = self._run_child("wedged-os-exit")
        self.assertNotIn("os._exit reached", text,
                         "TerminateProcess did not land; it fell through to the "
                         "os._exit that wedged the v2.0.191 gate")
        self.assertEqual(rc, tw.EXIT_CODE, text[-2000:])


class ClockIsolationTests(_WatchdogBase):
    """tests/_monolith_harness.shift_monotonic moves time.monotonic 30 days
    ahead for a test. The 2026-10-14 gate log shows the watchdog reading that
    patched clock: 'phase exceeded 120s (2.592e+06 elapsed)' on a test that
    had only just started."""

    def test_a_test_that_shifts_the_clock_cannot_trip_the_watchdog(self):
        stop = threading.Event()
        tw._state.update({"stop": stop, "per_phase": 120.0, "total": 900.0,
                          "run_start": time.monotonic()})
        tw.note("test tests.x.ShiftedClockTests.test_y")
        real = time.monotonic
        with mock.patch.object(time, "monotonic",
                               lambda: real() + 30 * 86400.0):
            self.assertIsNone(tw._blown(stop))

    def test_a_patched_sleep_cannot_shrink_the_grace_window(self):
        with mock.patch.object(time, "sleep") as patched_sleep, \
             mock.patch.object(tw, "DUMP_GRACE_S", 0.0), \
             mock.patch.object(tw, "_exit", lambda code: None), \
             mock.patch.object(tw, "_dump"), mock.patch.object(tw, "_write"):
            tw._fire("phase exceeded 1s: test x", lambda: True)
        patched_sleep.assert_not_called()


# ── arm / disarm ────────────────────────────────────────────────────────────
class ArmTests(_WatchdogBase):
    def test_arm_starts_a_daemon_thread_and_disarm_stops_it(self):
        with mock.patch("builtins.print"):
            self.assertTrue(tw.arm(per_phase_s=60.0, total_s=600.0))
        t = tw._state["thread"]
        self.assertIsNotNone(t)
        self.assertTrue(t.daemon, "the watchdog must never hold the process open")
        self.assertTrue(t.is_alive())
        tw.disarm()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())

    def test_arm_is_idempotent(self):
        with mock.patch("builtins.print"):
            tw.arm(per_phase_s=60.0, total_s=600.0)
            first = tw._state["thread"]
            tw.arm(per_phase_s=1.0, total_s=1.0)
        self.assertIs(tw._state["thread"], first,
                      "a second arm() must not start a second watchdog")

    def test_env_override_beats_the_callers_ceiling(self):
        with mock.patch.dict(os.environ, {tw.PER_PHASE_ENV: "7"}), \
             mock.patch("builtins.print"):
            tw.arm(per_phase_s=60.0, total_s=600.0)
        self.assertEqual(tw._state["per_phase"], 7.0)

    def test_fully_disabled_arms_nothing_and_says_so(self):
        with mock.patch.dict(os.environ, {tw.PER_PHASE_ENV: "off",
                                          tw.TOTAL_ENV: "off"}), \
             mock.patch("builtins.print") as pr:
            self.assertFalse(tw.arm())
        self.assertIsNone(tw._state["thread"])
        said = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("DISABLED", said)

    def test_disarm_without_arm_is_harmless(self):
        tw.disarm()
        tw.disarm()

    def test_a_superseded_watchdog_thread_never_fires(self):
        """Two watchdogs can briefly coexist (this very file swaps the state
        cell out from under the live one). Only the CURRENT one may kill the
        run — and the stale one must keep looping rather than exit, or a swap
        would silently leave the rest of the run unbounded."""
        fired: list = []
        stale_stop = threading.Event()
        self.addCleanup(stale_stop.set)
        tw._state["per_phase"] = 0.001      # long since blown
        tw._state["total"] = 0.0
        tw._cell[1] = time.monotonic() - 60
        with mock.patch.object(tw, "_fire", fired.append):
            t = threading.Thread(target=tw._watch, args=(stale_stop,),
                                 daemon=True)
            t.start()
            time.sleep(tw._POLL_S * 4)
            self.assertEqual(fired, [],
                             "a watchdog that does not own _state['stop'] fired")
            self.assertTrue(t.is_alive(), "the stale watchdog exited instead of "
                                          "waiting to become current again")
        stale_stop.set()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "disarm's event must still end it")

    def test_an_armed_watchdog_actually_fires(self):
        # End to end on the real thread, with a 0.05s budget and an injected
        # exit: proves the poll loop reaches _fire, not just that _expired
        # returns a string.
        fired = threading.Event()
        with mock.patch.object(tw, "_exit", lambda code: fired.set()), \
             mock.patch.object(tw, "DUMP_GRACE_S", 0.0), \
             mock.patch.object(tw, "_dump"), \
             mock.patch.object(tw, "_write"), \
             mock.patch("builtins.print"):
            tw.arm(per_phase_s=0.05, total_s=0.0)
            self.assertTrue(fired.wait(timeout=10),
                            "the watchdog thread never fired on a blown budget")


# ── the unittest wiring ─────────────────────────────────────────────────────
class ResultClassTests(_WatchdogBase):
    def test_the_result_class_labels_the_running_test_and_the_gap_after_it(self):
        seen: list = []
        # Built the way a runner builds it, not by hand: Python 3.14's
        # TextTestResult asks the stream for a real fileno() to pick a colour
        # theme, so a Mock stream errors out before the test even starts.
        runner = unittest.TextTestRunner(
            stream=io.StringIO(), resultclass=tw.WatchdogTextTestResult)
        res = runner._makeResult()
        self.assertIsInstance(res, tw.WatchdogTextTestResult)
        with mock.patch.object(tw, "note", seen.append):
            case = ResultClassTests("test_the_result_class_is_a_text_result")
            res.startTest(case)
            res.stopTest(case)
        self.assertEqual(len(seen), 2, seen)
        self.assertIn(case.id(), seen[0])
        self.assertTrue(seen[0].startswith("test "))
        self.assertTrue(seen[1].startswith("after test "),
                        "the gap after a test is its own phase: tearDown, "
                        "tearDownModule and the next module's import all live "
                        "there, and any of them can hang")

    def test_the_result_class_is_a_text_result(self):
        # It has to stay drop-in for TextTestRunner(resultclass=...), or the
        # runners silently lose their normal output.
        self.assertTrue(issubclass(tw.WatchdogTextTestResult,
                                   unittest.TextTestResult))

    def test_note_is_cheap_enough_to_call_around_every_test(self):
        t0 = time.monotonic()
        for _ in range(20_000):
            tw.note("test x")
        self.assertLess(time.monotonic() - t0, 1.0)


# ── the structural half: no runner may lose the watchdog ────────────────────
class RunnerWiringTests(unittest.TestCase):
    """The scan that stops the guard being dropped in a refactor — the exact
    failure that left the 2026-07-12 shutdown time-box in one copy while the
    other went on hanging the suite."""

    @staticmethod
    def _discovery_runners() -> list[str]:
        found = []
        for name in sorted(os.listdir(_TOOLS_DIR)):
            if not name.endswith(".py"):
                continue
            src = _source(os.path.join(_TOOLS_DIR, name))
            if any(marker in src for marker in _DISCOVERY_MARKERS):
                found.append(name)
        return found

    def test_the_guarded_list_is_exactly_the_discovery_runners(self):
        self.assertEqual(sorted(self._discovery_runners()),
                         sorted(_GUARDED_RUNNERS),
                         "a tools/ module started or stopped discovering the "
                         "suite — update _GUARDED_RUNNERS (and the twin list "
                         "in tests/test_mem_guard.py) or it stops being checked")

    def test_every_runner_arms_the_watchdog(self):
        for name in _GUARDED_RUNNERS:
            with self.subTest(runner=name):
                src = _source(os.path.join(_TOOLS_DIR, name))
                self.assertIn("from tools import test_watchdog", src)
                self.assertIn("test_watchdog.arm(", src,
                              f"{name} runs the suite with NO time ceiling")

    def test_every_runner_reports_which_test_is_running(self):
        """arm() alone bounds the run; the result class is what makes the
        report name a TEST instead of just 'somewhere in the suite'."""
        for name in _GUARDED_RUNNERS:
            with self.subTest(runner=name):
                src = _source(os.path.join(_TOOLS_DIR, name))
                self.assertIn("WatchdogTextTestResult", src)

    def test_every_runner_stands_the_watchdog_down_afterwards(self):
        for name in _GUARDED_RUNNERS:
            with self.subTest(runner=name):
                src = _source(os.path.join(_TOOLS_DIR, name))
                self.assertIn("test_watchdog.disarm()", src,
                              f"{name} leaves the watchdog armed over its own "
                              "summary, so a slow tail reads as a hang")

    def test_the_time_ceiling_is_armed_after_the_memory_ceiling(self):
        """Order is a contract: the memory ceiling must still be the first
        thing main() does (a runaway allocation bugchecked this box once), and
        the time ceiling goes in right behind it — before discovery, so an
        import-time hang is named too."""
        for name in _GUARDED_RUNNERS:
            with self.subTest(runner=name):
                src = _source(os.path.join(_TOOLS_DIR, name))
                self.assertLess(src.index("apply_memory_ceiling()"),
                                src.index("test_watchdog.arm("))

    def test_the_watchdog_is_armed_before_the_suite_is_loaded(self):
        """A hang during discovery is the invisible one — no test has started,
        so there is nothing in the output to name. arm() has to precede the
        loader in main()."""
        touching = {"discover", "loadTestsFromName", "loadTestsFromModule",
                    "_run_ci_gates", "_run_suite", "run", "Popen"}
        for name in _GUARDED_RUNNERS:
            with self.subTest(runner=name):
                path = os.path.join(_TOOLS_DIR, name)
                tree = ast.parse(_source(path))
                main = next(n for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == "main")
                arm_line = first_touch = None
                for node in ast.walk(main):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    attr = getattr(func, "attr", None) or getattr(func, "id", None)
                    if attr == "arm" and arm_line is None:
                        arm_line = node.lineno
                    elif attr in touching and (first_touch is None
                                               or node.lineno < first_touch):
                        first_touch = node.lineno
                self.assertIsNotNone(arm_line, f"{name}: main() never arms it")
                self.assertIsNotNone(first_touch, f"{name}: main() runs nothing?")
                self.assertLess(arm_line, first_touch,
                                f"{name}: the watchdog is armed at line "
                                f"{arm_line}, AFTER the suite is touched at "
                                f"line {first_touch}")


if __name__ == "__main__":
    unittest.main()
