"""Tests for tools/gate_run.py — the parent that makes sure a gate ENDS.

2026-10-13: the v2.0.191 gate's watchdog printed "[test-watchdog] exiting 3"
and the python.exe lived ~26 more hours while its wrapper waited on it. These
run real child interpreters (a parent that only ever met a mock would prove
nothing about killing one) with budgets of a second or two, so the whole file
costs a few seconds.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

from tools import gate_run
from tools import test_watchdog as tw

_HANG = "import threading; threading.Event().wait()"


class _GateRunBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="gate_run_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = os.path.join(self.dir, "gate.log")

    def _run(self, code: str, **kw) -> tuple[int, str, float]:
        t0 = time.monotonic()
        rc = gate_run.run(self.log, [sys.executable, "-c", code], **kw)
        with open(self.log, "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
        return rc, text, time.monotonic() - t0


class ChildOutcomeTests(_GateRunBase):
    def test_a_clean_child_passes_its_output_and_exit_code_through(self):
        rc, text, _ = self._run("print('hello from the gate'); raise SystemExit(7)",
                                max_seconds=60)
        self.assertEqual(rc, 7)
        self.assertIn("hello from the gate", text)
        self.assertTrue(text.rstrip().endswith("EXIT=7"),
                        "the wrapper's EXIT= line is the contract every gate "
                        "reader greps for")

    def test_a_child_that_says_exiting_but_lives_is_killed(self):
        """The 2026-10-13 shape, end to end: the watchdog line is in the log,
        the process is not gone. The parent must end it and report the
        watchdog's own exit code."""
        code = ("import os, threading; "
                "os.write(1, b'[test-watchdog] exiting 3 via os._exit\\n'); "
                "threading.Event().wait()")
        rc, text, took = self._run(code, max_seconds=60, exit_grace=0.5)
        self.assertEqual(rc, tw.EXIT_CODE)
        self.assertLess(took, 30, "the kill waited for the ceiling instead "
                                  "of acting on the watchdog's marker")
        self.assertIn("[gate-run] the test watchdog said it was exiting", text)
        self.assertTrue(text.rstrip().endswith("EXIT=%d" % tw.EXIT_CODE))

    def test_the_wall_clock_ceiling_ends_a_silent_hang(self):
        # The watchdog never fired at all (starved, or the GIL never came back):
        # only the parent's own clock can end this one.
        rc, text, took = self._run(_HANG, max_seconds=1.0)
        self.assertEqual(rc, gate_run.TIMEOUT_EXIT_CODE)
        self.assertLess(took, 30)
        self.assertIn("[gate-run] wall-clock ceiling of 1s blown", text)

    def test_the_child_runs_with_cuda_hidden(self):
        rc, text, _ = self._run(
            "import os; print('CUDA=' + os.environ.get('CUDA_VISIBLE_DEVICES', '?'))",
            max_seconds=60)
        self.assertEqual(rc, 0)
        self.assertIn("CUDA=-1", text)

    @unittest.skipUnless(os.name == "nt", "priority classes are Windows-only")
    def test_the_child_runs_at_idle_priority(self):
        code = ("import ctypes; k = ctypes.WinDLL('kernel32'); "
                "k.GetCurrentProcess.restype = ctypes.c_void_p; "
                "k.GetPriorityClass.argtypes = [ctypes.c_void_p]; "
                "print('PRIO=%#x' % k.GetPriorityClass(k.GetCurrentProcess()))")
        rc, text, _ = self._run(code, max_seconds=60)
        self.assertEqual(rc, 0, text)
        self.assertIn("PRIO=0x40", text)       # IDLE_PRIORITY_CLASS


class MarkerScanTests(_GateRunBase):
    def test_a_marker_split_across_two_reads_is_found(self):
        scan = gate_run._MarkerScan(self.log)
        half = len(gate_run.MARKER) // 2
        with open(self.log, "wb") as fh:
            fh.write(b"....." + gate_run.MARKER[:half])
        self.assertFalse(scan.seen())
        with open(self.log, "ab") as fh:
            fh.write(gate_run.MARKER[half:] + b" 3\n")
        self.assertTrue(scan.seen())

    def test_a_missing_log_is_not_a_marker(self):
        self.assertFalse(gate_run._MarkerScan(
            os.path.join(self.dir, "nope.log")).seen())


class CeilingTests(unittest.TestCase):
    def test_the_default_ceiling_clears_the_slowest_runner(self):
        # run_coverage.py arms the watchdog at 4x the total budget; a ceiling
        # under that would kill a legal coverage run.
        with mock.patch.dict(os.environ, {tw.TOTAL_ENV: "1800"}):
            self.assertGreater(gate_run.default_ceiling(), 4 * 1800)

    def test_a_disabled_total_budget_disables_the_default_ceiling(self):
        with mock.patch.dict(os.environ, {tw.TOTAL_ENV: "off"}):
            self.assertEqual(gate_run.default_ceiling(), 0.0)


class CliTests(_GateRunBase):
    def test_main_without_a_command_is_a_usage_error(self):
        with mock.patch("sys.stderr"):
            self.assertEqual(gate_run.main([self.log]), 2)

    def test_main_runs_the_command_after_the_separator(self):
        with mock.patch("builtins.print"):
            rc = gate_run.main([self.log, "--max-seconds", "60", "--",
                                sys.executable, "-c", "raise SystemExit(5)"])
        self.assertEqual(rc, 5)


if __name__ == "__main__":
    unittest.main()
