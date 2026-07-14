"""tools/jarvis_watchdog.py — the resurrection net's liveness check.

CORPSE BLINDNESS (2026-07-14, live): _jarvis_running() COUNTED the CIM rows
whose command line matched bobert_companion and called any count > 0 "alive".
A kernel-stuck 'terminating forever' process (thread parked in a CUDA/audio
driver at exit) keeps its row — command line intact — until Windows reboots.
So one corpse permanently convinced the watchdog that JARVIS was running: the
real JARVIS died at 10:49 and every 5-minute tick no-opped against two
day-old corpses. The check must ask whether each PID is GENUINELY EXECUTING.
"""
import importlib.util
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

_spec = importlib.util.spec_from_file_location(
    "jarvis_watchdog_under_test",
    os.path.join(_PROJECT, "tools", "jarvis_watchdog.py"))
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)


def _cim(pids):
    """Fake the PowerShell CIM query: it prints one PID per line."""
    out = mock.Mock()
    out.stdout = "\n".join(str(p) for p in pids)
    return out


class JarvisRunningTests(unittest.TestCase):
    def test_no_rows_means_dead(self):
        with mock.patch.object(wd.subprocess, "run", return_value=_cim([])):
            self.assertFalse(wd._jarvis_running())

    def test_live_pid_means_running(self):
        import core.parent_watch as pw
        with mock.patch.object(wd.subprocess, "run", return_value=_cim([4242])), \
             mock.patch.object(pw, "parent_is_alive", return_value=True):
            self.assertTrue(wd._jarvis_running())

    def test_corpse_only_means_DEAD(self):
        # THE regression: rows exist, but every one of them is a kernel-stuck
        # corpse. The watchdog must resurrect, not sit on its hands.
        import core.parent_watch as pw
        with mock.patch.object(wd.subprocess, "run",
                               return_value=_cim([50916, 53452])), \
             mock.patch.object(pw, "parent_is_alive", return_value=False), \
             mock.patch.object(wd, "_note") as note:
            self.assertFalse(wd._jarvis_running())
        # and it says so in the log, so the next human knows why it booted
        self.assertTrue(any("CORPSE" in str(c) for c in note.call_args_list))

    def test_live_pid_beside_corpses_means_running(self):
        # A healthy instance next to yesterday's corpses must NOT be double-booted.
        import core.parent_watch as pw
        with mock.patch.object(wd.subprocess, "run",
                               return_value=_cim([50916, 7777])), \
             mock.patch.object(pw, "parent_is_alive",
                               side_effect=lambda p: p == 7777):
            self.assertTrue(wd._jarvis_running())

    def test_query_failure_fails_safe(self):
        # An unreadable process table must never cause a double boot.
        with mock.patch.object(wd.subprocess, "run",
                               side_effect=OSError("wmi down")):
            self.assertTrue(wd._jarvis_running())


if __name__ == "__main__":
    unittest.main()
