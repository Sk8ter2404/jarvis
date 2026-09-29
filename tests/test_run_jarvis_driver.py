"""run-jarvis driver (.claude/skills/run-jarvis/driver.py) liveness + tray channel.

2026-09-29 14:57, mid test-sweep: an idle JARVIS only writes its session log
('Listening…') about every 20 s, the driver's is_running() demanded a write within
15 s, decided JARVIS was down and ran _boot_jarvis.ps1 -- whose first step KILLS the
running instance. A healthy JARVIS was restarted by its own test harness.
is_running() now asks the process table first; log freshness is only the fallback
when processes can't be listed.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRIVER = os.path.join(os.path.dirname(_HERE), ".claude", "skills", "run-jarvis", "driver.py")


def _load_driver():
    spec = importlib.util.spec_from_file_location("_run_jarvis_driver_under_test", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Proc:
    def __init__(self, pid, name, cmdline):
        self.info = {"pid": pid, "name": name, "cmdline": cmdline}


def _fake_psutil(procs):
    m = types.ModuleType("psutil")
    m.process_iter = lambda attrs=None: iter(procs)
    return m


@unittest.skipUnless(os.path.exists(_DRIVER), "run-jarvis driver not present")
class DriverLivenessTests(unittest.TestCase):
    def setUp(self):
        self.d = _load_driver()
        self.tmp = tempfile.mkdtemp(prefix="drv_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        logs = os.path.join(self.tmp, "logs")
        os.makedirs(logs)
        for name, val in (("PROJ", self.tmp), ("LOGS", logs),
                          ("TRAY", os.path.join(self.tmp, "tray_commands.json")),
                          ("INJECT", os.path.join(self.tmp, "injected_commands.json")),
                          ("BOOT", os.path.join(self.tmp, "_boot_jarvis.ps1"))):
            p = mock.patch.object(self.d, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.log = os.path.join(logs, "session_2026-09-29_14-00-00.log")
        with open(self.log, "w", encoding="utf-8") as f:
            f.write("Listening...\n")

    def _age_log(self, seconds):
        t = time.time() - seconds
        os.utime(self.log, (t, t))

    def _psutil(self, procs):
        return mock.patch.dict(sys.modules, {"psutil": _fake_psutil(procs)})

    def test_idle_jarvis_with_a_stale_log_is_still_running(self):
        # The 14:57 regression: last log write 20 s ago, process alive.
        self._age_log(20)
        with self._psutil([_Proc(4242, "pythonw.exe", ["pythonw", "C:\\J\\bobert_companion.py"])]):
            self.assertTrue(self.d.is_running())

    def test_staging_instance_does_not_count(self):
        self._age_log(1)
        with self._psutil([_Proc(7, "python.exe", ["python", "bobert_companion.py", "--staging"])]):
            self.assertFalse(self.d.is_running())

    def test_no_process_means_not_running_even_with_a_fresh_log(self):
        self._age_log(1)
        with self._psutil([_Proc(9, "python.exe", ["python", "other_tool.py"]),
                           _Proc(10, "notepad.exe", ["notepad", "bobert_companion.py"])]):
            self.assertFalse(self.d.is_running())

    def test_falls_back_to_log_freshness_without_psutil(self):
        with mock.patch.object(self.d, "_prod_jarvis_pids", return_value=None):
            self._age_log(2)
            self.assertTrue(self.d.is_running())
            self._age_log(self.d.ALIVE_WINDOW_S + 30)
            self.assertFalse(self.d.is_running())

    def test_main_never_boots_over_a_live_process(self):
        self._age_log(60)
        boot = mock.Mock(return_value=True)
        with self._psutil([_Proc(4242, "python.exe", ["python", "bobert_companion.py"])]), \
                mock.patch.object(self.d, "boot", boot), \
                mock.patch.object(self.d.time, "sleep"), \
                mock.patch.object(sys, "argv", ["driver.py", "--wake"]):
            self.assertEqual(self.d.main(), 0)
        boot.assert_not_called()

    def test_force_wake_appends_to_pending_tray_commands(self):
        with open(self.d.TRAY, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "restart"}], f)
        with mock.patch.object(self.d.time, "sleep"):
            self.d.force_wake()
        with open(self.d.TRAY, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [{"cmd": "restart"}, {"cmd": "force_wake"}])

    def test_force_wake_recovers_from_a_corrupt_tray_file(self):
        with open(self.d.TRAY, "w", encoding="utf-8") as f:
            f.write("{not json")
        with mock.patch.object(self.d.time, "sleep"):
            self.d.force_wake()
        with open(self.d.TRAY, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [{"cmd": "force_wake"}])


if __name__ == "__main__":
    unittest.main()
