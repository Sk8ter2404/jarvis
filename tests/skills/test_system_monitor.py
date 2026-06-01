"""Logic tests for skills/system_monitor.py.

The check_system action builds a JARVIS-cadence CPU/RAM/disk/network report
from psutil. We pin every psutil source (cpu_percent, virtual_memory,
disk_usage) and stub the two helper collectors (_top_processes, _network_rates)
so the rendered sentence is deterministic and no live machine probe runs.

The background _monitor_loop is never started (harness neuters threads); we
test _build_report and the graceful psutil-missing degradation instead.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _VM:
    """Minimal stand-in for psutil.virtual_memory()."""
    def __init__(self, percent, used_gb, total_gb=32.0):
        self.percent = percent
        self.used = int(used_gb * 1024 ** 3)
        self.total = int(total_gb * 1024 ** 3)


class _Disk:
    def __init__(self, free_gb, total_gb):
        self.free = int(free_gb * 1024 ** 3)
        self.total = int(total_gb * 1024 ** 3)


class SystemMonitorReportTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")

    def _run_report(self, *, cpu, ram_pct, ram_used=8.0, top=None,
                    net=(0.0, 0.0), disk=(500.0, 1000.0)):
        top = top if top is not None else []
        fake_psutil = mock.MagicMock()
        fake_psutil.cpu_percent.return_value = cpu
        fake_psutil.virtual_memory.return_value = _VM(ram_pct, ram_used)
        fake_psutil.disk_usage.return_value = _Disk(*disk)
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake_psutil), \
             mock.patch.object(self.mod, "_top_processes", return_value=top), \
             mock.patch.object(self.mod, "_network_rates", return_value=net):
            return self.mod._build_report()

    def test_nominal_opener(self):
        out = self._run_report(cpu=10, ram_pct=40)
        self.assertTrue(out.startswith("Systems nominal, sir."))
        self.assertIn("CPU at 10 percent", out)

    def test_holding_opener(self):
        out = self._run_report(cpu=70, ram_pct=85)
        self.assertTrue(out.startswith("Systems holding up, sir."))

    def test_working_hard_opener(self):
        out = self._run_report(cpu=95, ram_pct=92)
        self.assertIn("working rather hard", out)

    def test_chrome_offender_phrasing(self):
        out = self._run_report(cpu=50, ram_pct=50,
                               top=[("chrome.exe", 88.0)])
        self.assertIn("Chrome is, as usual, the primary offender", out)

    def test_generic_offender_phrasing(self):
        out = self._run_report(cpu=50, ram_pct=50,
                               top=[("blender.exe", 70.0)])
        self.assertIn("blender.exe is the primary offender", out)

    def test_disk_line_included(self):
        out = self._run_report(cpu=10, ram_pct=30, disk=(123.0, 1000.0))
        self.assertIn("C drive has 123 gigs free of 1000", out)

    def test_network_line_included_when_active(self):
        out = self._run_report(cpu=10, ram_pct=30, net=(2048.0, 512.0))
        self.assertIn("network at 2048 down, 512 up", out)

    def test_network_line_omitted_when_quiet(self):
        out = self._run_report(cpu=10, ram_pct=30, net=(1.0, 1.0))  # < 5 kbps
        self.assertNotIn("network at", out)

    def test_ram_committed_reported(self):
        out = self._run_report(cpu=10, ram_pct=30, ram_used=12.0)
        self.assertIn("12 of 32 gigs committed", out)


class SystemMonitorDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")

    def test_build_report_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            out = self.mod._build_report()
        self.assertIn("requires the psutil package", out)

    def test_check_system_action_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            out = self.actions["check_system"]("")
        self.assertIn("psutil", out.lower())

    def test_check_system_wraps_exceptions(self):
        with mock.patch.object(self.mod, "_build_report",
                               side_effect=RuntimeError("boom")):
            out = self.actions["check_system"]("")
        self.assertIn("system check failed", out.lower())
        self.assertIn("boom", out)

    def test_top_processes_empty_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._top_processes(3), [])

    def test_network_rates_zero_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._network_rates(), (0.0, 0.0))


class SystemMonitorTopProcessTests(unittest.TestCase):
    """_top_processes aggregates per-name CPU and drops the idle process."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")

    def test_aggregates_and_sorts_and_drops_idle(self):
        # Two chrome children + one idle process. cpu_percent is called twice
        # per proc (throwaway then real); we return the "real" value on the
        # second read for each.
        def _proc(name, cpu):
            p = mock.MagicMock()
            p.info = {"name": name}
            p.cpu_percent.side_effect = [0.0, cpu]
            return p

        procs = [_proc("chrome.exe", 30.0), _proc("chrome.exe", 25.0),
                 _proc("System Idle Process", 99.0), _proc("python.exe", 10.0)]
        fake_psutil = mock.MagicMock()
        fake_psutil.process_iter.return_value = procs
        # psutil.NoSuchProcess / AccessDenied must be real exception classes.
        fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake_psutil), \
             mock.patch.object(self.mod.time, "sleep"):
            top = self.mod._top_processes(3)
        names = [n for n, _ in top]
        self.assertNotIn("System Idle Process", names)
        # chrome aggregated to 55.0, ahead of python at 10.0
        self.assertEqual(top[0][0], "chrome.exe")
        self.assertAlmostEqual(top[0][1], 55.0)


if __name__ == "__main__":
    unittest.main()
