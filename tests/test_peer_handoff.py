"""core.peer_handoff - one JARVIS at a time across the owner's machines
(2026-10-19: the laptop's brain is the desktop's GPU, so two running copies
fight over one card and both answer the wake word).

No real ssh, WMI or process is ever touched: every runner is a fake.

    python -m unittest tests.test_peer_handoff
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from core import config as cfg
from core import peer_handoff as H


def _ran(stdout="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, returncode=returncode)


class _Cfg(unittest.TestCase):
    def setUp(self):
        for key, val in (("PEER_HANDOFF_SSH", "brade@peer-host"),
                         ("PEER_HANDOFF_NAME", "laptop"),
                         ("HANDOFF_RESUME_WHEN_PEER_STOPS", False)):
            p = mock.patch.object(cfg, key, val, create=True)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {"JARVIS_STAGING": "", "JARVIS_HEADLESS": ""})
        env.start()
        self.addCleanup(env.stop)


class SwitchTests(_Cfg):
    def test_off_without_a_peer(self):
        with mock.patch.object(cfg, "PEER_HANDOFF_SSH", ""):
            self.assertFalse(H.enabled())
            self.assertEqual(H.ask_peer_to_stand_down(run=mock.Mock(), log=lambda m: None), "off")

    def test_on_with_a_peer(self):
        self.assertTrue(H.enabled())
        self.assertEqual(H.peer_target(), "brade@peer-host")

    def test_staging_and_headless_never_take_part(self):
        for var in ("JARVIS_STAGING", "JARVIS_HEADLESS"):
            with self.subTest(var=var), mock.patch.dict(os.environ, {var: "1"}):
                self.assertFalse(H.enabled())

    def test_a_target_that_is_not_user_at_host_is_refused(self):
        for bad in ("brade@host; calc", "-oProxyCommand=x", "host", "a b@c"):
            with self.subTest(bad=bad), mock.patch.object(cfg, "PEER_HANDOFF_SSH", bad):
                self.assertEqual(H.peer_target(), "")
                self.assertFalse(H.enabled())


class AskPeerTests(_Cfg):
    def test_the_peer_runs_stand_down_with_a_clean_from_name(self):
        run = mock.Mock(return_value=_ran("stood down\n"))
        with mock.patch.object(H.socket, "gethostname", return_value='DESK"; x'):
            out = H.ask_peer_to_stand_down(run=run, log=lambda m: None)
        self.assertEqual(out, "stood down")
        argv = run.call_args[0][0]
        self.assertEqual(argv[:2], ["ssh", "-n"])
        self.assertIn("BatchMode=yes", argv)
        self.assertEqual(argv[-2], "brade@peer-host")
        self.assertIn("stand_down.ps1", argv[-1])
        self.assertIn('-From "DESK x"', argv[-1])

    def test_an_unreachable_peer_is_normal(self):
        log = []
        out = H.ask_peer_to_stand_down(run=mock.Mock(return_value=_ran("", 255)), log=log.append)
        self.assertEqual(out, "unreachable (exit 255)")
        self.assertIn("laptop", log[0])
        out = H.ask_peer_to_stand_down(
            run=mock.Mock(side_effect=subprocess.TimeoutExpired("ssh", 75)), log=lambda m: None)
        self.assertEqual(out, "unreachable (TimeoutExpired)")


class RequestTests(_Cfg):
    def test_only_a_request_after_this_start_counts(self):
        self.assertTrue(H.request_is_for_me({"at": 200.0}, 100.0))
        self.assertFalse(H.request_is_for_me({"at": 50.0}, 100.0))
        self.assertFalse(H.request_is_for_me({"at": "junk"}, 100.0))
        self.assertFalse(H.request_is_for_me({}, 100.0))

    def test_goodbye_names_the_machine_only_for_a_handoff(self):
        self.assertEqual(H.goodbye_for(H.handoff_reason("laptop")),
                         "Handing over to the laptop, sir.")
        self.assertEqual(H.goodbye_for("handoff:"), "Handing over to the other machine, sir.")
        self.assertEqual(H.goodbye_for(""), "")
        self.assertEqual(H.goodbye_for("shut down"), "")
        self.assertEqual(H.goodbye_for(None), "")


class WatchTests(_Cfg):
    def setUp(self):
        super().setUp()
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "stand_down.json")

    def tearDown(self):
        try:
            os.remove(self.path)
        except OSError:
            pass
        os.rmdir(self.dir)

    def _write(self, at, who="DESK"):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"from": who, "at": at}, f)

    def _watch(self, started_at, resume_on=False, timeout=1.0):
        calls, resumed, stop = [], [], threading.Event()
        with mock.patch.object(cfg, "HANDOFF_RESUME_WHEN_PEER_STOPS", resume_on):
            t = threading.Thread(target=H.watch, args=(started_at, calls.append, stop),
                                 kwargs={"path": self.path, "poll_s": 0.01,
                                         "resume": lambda: resumed.append(1),
                                         "log": lambda m: None})
            t.start()
            t.join(timeout)
            stop.set()
            t.join(1.0)
        return calls, resumed

    def test_a_fresh_request_stands_this_copy_down(self):
        self._write(time.time() + 5)
        calls, resumed = self._watch(time.time())
        self.assertEqual(calls, ["handoff:laptop"])
        self.assertEqual(resumed, [])

    def test_with_resume_on_the_waiter_starts_before_the_shutdown(self):
        self._write(time.time() + 5)
        calls, resumed = self._watch(time.time(), resume_on=True)
        self.assertEqual((calls, resumed), (["handoff:laptop"], [1]))

    def test_a_request_from_before_this_boot_is_ignored(self):
        self._write(time.time() - 600)
        calls, resumed = self._watch(time.time(), resume_on=True, timeout=0.15)
        self.assertEqual((calls, resumed), ([], []))

    def test_no_request_file_is_quiet(self):
        calls, _ = self._watch(time.time(), timeout=0.1)
        self.assertEqual(calls, [])


class ResumeWaiterTests(_Cfg):
    def test_the_waiter_goes_through_wmi_with_the_peer_and_this_pid(self):
        run = mock.Mock(return_value=_ran("0\n"))
        with mock.patch.object(H.os.path, "exists", return_value=True):
            self.assertTrue(H.launch_resume_waiter(run=run, log=lambda m: None))
        ps = run.call_args[0][0][-1]
        self.assertIn("Invoke-CimMethod -ClassName Win32_Process -MethodName Create", ps)
        self.assertIn("handoff_resume.ps1", ps)
        self.assertIn("-Peer brade@peer-host", ps)
        self.assertIn(f"-StoodDownPid {os.getpid()}", ps)

    def test_a_failed_wmi_create_is_reported(self):
        log = []
        with mock.patch.object(H.os.path, "exists", return_value=True):
            self.assertFalse(H.launch_resume_waiter(run=mock.Mock(return_value=_ran("9\n")),
                                                    log=log.append))
        self.assertIn("FAILED", log[0])

    def test_no_peer_no_waiter(self):
        run = mock.Mock()
        with mock.patch.object(cfg, "PEER_HANDOFF_SSH", ""):
            self.assertFalse(H.launch_resume_waiter(run=run, log=lambda m: None))
        run.assert_not_called()


class StartTests(_Cfg):
    def test_start_is_a_no_op_when_off(self):
        with mock.patch.object(cfg, "PEER_HANDOFF_SSH", ""), \
                mock.patch.object(H.threading, "Thread") as thread:
            self.assertFalse(H.start(lambda r: None, log=lambda m: None))
        thread.assert_not_called()

    def test_start_runs_the_ask_and_the_watch_in_the_background(self):
        with mock.patch.object(H.threading, "Thread") as thread:
            self.assertTrue(H.start(lambda r: None, log=lambda m: None))
        names = [c.kwargs.get("name") for c in thread.call_args_list]
        self.assertEqual(names, ["peer-handoff-ask", "peer-handoff-watch"])
        self.assertTrue(all(c.kwargs.get("daemon") for c in thread.call_args_list))


if __name__ == "__main__":
    unittest.main()
