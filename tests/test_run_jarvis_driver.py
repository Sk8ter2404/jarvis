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

    def _main_with(self, argv, results):
        fw, inj = mock.Mock(), mock.Mock()
        waits = iter(results)
        with self._psutil([_Proc(4242, "python.exe", ["python", "bobert_companion.py"])]),                 mock.patch.object(self.d, "force_wake", fw),                 mock.patch.object(self.d, "inject", inj),                 mock.patch.object(self.d, "wait_for_reply", lambda *a, **k: next(waits)),                 mock.patch.object(sys, "argv", ["driver.py"] + argv),                 mock.patch("builtins.print"):
            rc = self.d.main()
        return rc, fw, inj

    def test_driving_a_turn_does_not_wake_an_awake_jarvis(self):
        # Live 2026-09-29: an unconditional force_wake made JARVIS say "At your
        # service, sir." before every driven turn; his mic heard it and he
        # answered himself.
        rc, fw, inj = self._main_with(["what time is it"],
                                      [{"status": "ok", "lines": ["JARVIS: 3 PM"]}])
        self.assertEqual(rc, 0)
        fw.assert_not_called()
        inj.assert_called_once_with("what time is it")

    def test_standby_drop_wakes_once_and_retries(self):
        rc, fw, inj = self._main_with(["what time is it"],
                                      [{"status": "standby_ignored", "lines": []},
                                       {"status": "ok", "lines": ["JARVIS: 3 PM"]}])
        self.assertEqual(rc, 0)
        fw.assert_called_once()
        self.assertEqual(inj.call_count, 2)

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


@unittest.skipUnless(os.path.exists(_DRIVER), "run-jarvis driver not present")
class DriverReplyCaptureTests(unittest.TestCase):
    """The quality sweep of 2026-09-29 lost every informative action's real
    answer: the driver returned 2 s after the [action] line, but the follow-up
    round prints 'Reading results' up to ~4 s later and answers after that.
    And a plain reply with no action sat out the full 75 s timeout."""

    def setUp(self):
        self.d = _load_driver()

    def test_plain_reply_completes_after_quiet(self):
        lines = ["[t]   JARVIS: Canberra, sir."]
        self.assertFalse(self.d._reply_complete(lines, False, 1.0))
        self.assertTrue(self.d._reply_complete(lines, False, self.d.QUIET_S))

    def test_action_tag_waits_for_its_result(self):
        lines = ["[t]   JARVIS: [ACTION: get_time] One moment, sir."]
        self.assertFalse(self.d._reply_complete(lines, False, 60.0))
        lines.append("[t]   [action] get_time: current time is 03:21 PM")
        self.assertFalse(self.d._reply_complete(lines, False, self.d.QUIET_S))
        self.assertTrue(self.d._reply_complete(lines, False, self.d.ACTION_QUIET_S))

    def test_pending_followup_never_completes(self):
        lines = ["[t]   JARVIS: [ACTION: get_time] One moment, sir.",
                 "[t]   [action] get_time: current time is 03:21 PM"]
        self.assertFalse(self.d._reply_complete(lines, True, 600.0))

    def test_wait_for_reply_captures_the_followup_answer(self):
        tmp = tempfile.mkdtemp(prefix="drv_")
        self.addCleanup(shutil.rmtree, tmp, True)
        log = os.path.join(tmp, "session_x.log")
        open(log, "w", encoding="utf-8").close()
        script = [  # (fake seconds after start, line) -- as seen live
            (0.5, "[15:21:17]   [inject] what time is it"),
            (4.0, "[15:21:21]   JARVIS: [ACTION: get_time] One moment, sir."),
            (4.0, "[15:21:21]   [action] get_time: current time is 03:21 PM"),
            (8.0, "[15:21:25]   Reading results (depth 1)…"),
            (9.0, "[15:21:26]   JARVIS: It is 3:21 PM, sir."),
        ]
        clock = [1000.0]

        def fake_sleep(s):
            before = clock[0] - 1000.0
            clock[0] += s
            now = clock[0] - 1000.0
            with open(log, "a", encoding="utf-8") as f:
                for t, line in script:
                    if before < t <= now:
                        f.write(line + "\n")

        with mock.patch.object(self.d, "latest_log", return_value=log), \
                mock.patch.object(self.d.time, "sleep", fake_sleep), \
                mock.patch.object(self.d.time, "time", lambda: clock[0]):
            res = self.d.wait_for_reply("what time is it", timeout=75.0)
        self.assertEqual(res["status"], "ok")
        self.assertIn("It is 3:21 PM, sir.", res["lines"][-1])
        self.assertLess(clock[0] - 1000.0, 20.0)   # nowhere near the 75 s timeout

    def _replay(self, script, text, timeout=30.0):
        """wait_for_reply over a scripted log: (fake seconds, line) pairs."""
        tmp = tempfile.mkdtemp(prefix="drv_")
        self.addCleanup(shutil.rmtree, tmp, True)
        log = os.path.join(tmp, "session_x.log")
        open(log, "w", encoding="utf-8").close()
        clock = [1000.0]

        def fake_sleep(s):
            before = clock[0] - 1000.0
            clock[0] += s
            now = clock[0] - 1000.0
            with open(log, "a", encoding="utf-8") as f:
                for t, line in script:
                    if before < t <= now:
                        f.write(line + "\n")

        with mock.patch.object(self.d, "latest_log", return_value=log), \
                mock.patch.object(self.d.time, "sleep", fake_sleep), \
                mock.patch.object(self.d.time, "time", lambda: clock[0]):
            return self.d.wait_for_reply(text, timeout=timeout)

    def test_a_wake_led_standby_inject_runs_and_is_not_retried(self):
        # 2026-10-01: standby now RUNS a wake-led command. Reading every
        # "(standby)" inject as dropped force-woke and re-injected it, so the
        # command ran twice.
        res = self._replay([
            (0.5, "[09:00:01]   [inject] (standby) Jarvis, what time is it"),
            (0.5, "[09:00:01]   [wake] Waking up"),
            (0.5, "[09:00:01]   [wake] the wake carries a command \u2014 running it now"),
            (3.0, "[09:00:04]   JARVIS: It is nine o'clock, sir."),
        ], "Jarvis, what time is it")
        self.assertEqual(res["status"], "ok")
        self.assertIn("nine o'clock", res["lines"][-1])

    def test_a_bare_wake_inject_returns_the_greeting(self):
        res = self._replay([
            (0.5, "[09:00:01]   [inject] (standby) Jarvis"),
            (0.5, "[09:00:01]   [wake] Waking up"),
            (0.5, "[09:00:01]   [wake] greeting='Yes, sir?' vol=1.0"),
        ], "Jarvis")
        self.assertEqual(res["status"], "ok")
        self.assertIn("Yes, sir?", res["lines"][-1])

    def test_an_unprefixed_standby_inject_is_reported_dropped(self):
        # The drop line carries the length only now, never the words.
        res = self._replay([
            (0.5, "[09:00:01]   [inject] (standby) what time is it"),
            (0.5, "[09:00:01]   [standby] ignored (15 chars)"),
        ], "what time is it")
        self.assertEqual(res["status"], "standby_ignored")

    def test_wait_for_reply_keeps_the_spoken_fallback_line(self):
        # v2.0.136 logs "JARVIS (spoken): ..." when a fallback replaced the
        # model's reply; a sweep that only saw the "JARVIS:" line judged the
        # dodge that was never said (2026-09-29).
        tmp = tempfile.mkdtemp(prefix="drv_")
        self.addCleanup(shutil.rmtree, tmp, True)
        log = os.path.join(tmp, "session_x.log")
        open(log, "w", encoding="utf-8").close()
        script = [
            (0.5, "[20:18:36]   [inject] what should I have for dinner"),
            (3.0, "[20:18:39]   JARVIS: [intent:dry_wit] A bold choice, sir."),
            (3.0, "[20:18:39]   [advice-fallback] reply dodged a request"),
            (3.0, "[20:18:39]   JARVIS (spoken): A stir-fry, sir."),
        ]
        clock = [1000.0]

        def fake_sleep(s):
            before = clock[0] - 1000.0
            clock[0] += s
            now = clock[0] - 1000.0
            with open(log, "a", encoding="utf-8") as f:
                for t, line in script:
                    if before < t <= now:
                        f.write(line + "\n")

        with mock.patch.object(self.d, "latest_log", return_value=log), \
                mock.patch.object(self.d.time, "sleep", fake_sleep), \
                mock.patch.object(self.d.time, "time", lambda: clock[0]):
            res = self.d.wait_for_reply("what should I have for dinner",
                                        timeout=75.0)
        self.assertEqual(res["status"], "ok")
        self.assertIn("JARVIS (spoken): A stir-fry, sir.", res["lines"][-1])


@unittest.skipUnless(os.path.exists(_DRIVER), "run-jarvis driver not present")
class DriverInjectSourceTests(unittest.TestCase):
    """B023 (2026-10-01): a verification line this harness injects is not the
    owner speaking; it is tagged so JARVIS answers it but never learns it."""

    def test_inject_tags_its_lines_as_test(self):
        d = _load_driver()
        tmp = tempfile.mkdtemp(prefix="drv_")
        self.addCleanup(shutil.rmtree, tmp, True)
        inject = os.path.join(tmp, "injected_commands.json")
        with mock.patch.object(d, "PROJ", tmp), \
                mock.patch.object(d, "INJECT", inject):
            d.inject("what time is it")
            d.inject("and the date")
        with open(inject, encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual([i["text"] for i in items],
                         ["what time is it", "and the date"])
        self.assertEqual({i.get("source") for i in items}, {"test"})


if __name__ == "__main__":
    unittest.main()
