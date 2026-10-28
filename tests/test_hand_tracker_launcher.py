"""core/hand_tracker_launcher.py with a FAKE Popen: prerequisites, the exact
command line, start/stop, the lifeline file, restart limits. No subprocess."""
import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import config  # noqa: E402
from core import hand_tracker_launcher as htl  # noqa: E402


class FakeProc:
    def __init__(self, argv, **kw):
        self.argv = argv
        self.kw = kw
        self.pid = 4242
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 1


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        self.alive = os.path.join(d, ".handtrack_alive")
        self.life = os.path.join(d, ".handtrack_jarvis_alive")
        self.procs = []
        self.logs = []
        self.exist = {"py", "model", htl.SERVICE_SCRIPT}
        self.clock = [1000.0]
        p = mock.patch.multiple(config, HAND_TRACK_PYTHON="py",
                                HAND_TRACK_MODEL="model", HAND_TRACK_PORT=8793)
        p.start()
        self.addCleanup(p.stop)

        def popen(argv, **kw):
            pr = FakeProc(argv, **kw)
            self.procs.append(pr)
            return pr

        self.la = htl.HandTrackerLauncher(
            popen=popen, clock=lambda: self.clock[0], pid=777,
            log=self.logs.append, exists=lambda p: p in self.exist,
            alive_file=self.alive, jarvis_file=self.life,
            log_path=os.path.join(d, "ht.log"), start_keeper=False)

    def tearDown(self):
        self.dir.cleanup()

    def test_prerequisites_are_checked_first(self):
        self.exist.discard("py")
        ok, msg = self.la.start()
        self.assertFalse(ok)
        self.assertIn("Python environment", msg)
        self.assertEqual(self.procs, [])
        self.assertFalse(self.la.wanted())
        self.exist.add("py")
        self.exist.discard("model")
        self.assertIn("model", self.la.start()[1])
        self.exist.add("model")
        self.exist.discard(htl.SERVICE_SCRIPT)
        self.assertIn("script", self.la.start()[1])

    def test_command_line(self):
        ok, msg = self.la.start()
        self.assertEqual((ok, msg), (True, "started"))
        argv = self.procs[0].argv
        self.assertEqual(argv[0], "py")
        self.assertIn(htl.SERVICE_SCRIPT, argv)
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], "8793")
        self.assertEqual(argv[argv.index("--model") + 1], "model")
        self.assertEqual(argv[argv.index("--jarvis-pid") + 1], "777")
        self.assertEqual(argv[argv.index("--jarvis-alive") + 1], self.life)
        self.assertEqual(argv[argv.index("--alive") + 1], self.alive)
        kw = self.procs[0].kw
        self.assertEqual(kw["stdin"], __import__("subprocess").DEVNULL)
        self.assertNotIn("PYTHONPATH", kw["env"])
        # Guard on the real capability, not just os.name: ci-sim leaves os.name
        # "nt" but strips subprocess.CREATE_NO_WINDOW to look like Linux, where
        # the product's nt branch never runs at all.
        import subprocess
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            self.assertTrue(kw["creationflags"] & subprocess.CREATE_NO_WINDOW)
        self.assertTrue(os.path.exists(self.life))          # the lifeline exists

    def test_start_is_idempotent(self):
        self.la.start()
        ok, msg = self.la.start()
        self.assertEqual((ok, msg), (True, "already running"))
        self.assertEqual(len(self.procs), 1)

    def test_spawn_failure_is_reported_not_raised(self):
        def boom(argv, **kw):
            raise OSError("no such file")
        la = htl.HandTrackerLauncher(popen=boom, log=self.logs.append,
                                     exists=lambda p: p in self.exist,
                                     alive_file=self.alive, jarvis_file=self.life,
                                     log_path=os.path.join(self.dir.name, "x.log"),
                                     start_keeper=False)
        ok, msg = la.start()
        self.assertFalse(ok)
        self.assertIn("OSError", msg)
        self.assertFalse(la.wanted())

    def test_stop_removes_the_lifeline_and_terminates_a_lingerer(self):
        self.la.start()
        pr = self.procs[0]
        with mock.patch.object(htl.time, "sleep", lambda s: self.clock.__setitem__(
                0, self.clock[0] + 1.0)):
            self.la.stop()
        self.assertFalse(os.path.exists(self.life))
        self.assertTrue(pr.terminated)                       # lingered past the grace
        self.assertFalse(self.la.wanted())
        self.assertFalse(self.la.running())

    def test_stop_lets_a_service_that_exits_alone_go(self):
        self.la.start()
        pr = self.procs[0]
        pr.code = 0
        self.la.stop()
        self.assertFalse(pr.terminated)

    def test_tick_touches_the_lifeline_and_restarts_a_dead_service(self):
        self.la.start()
        os.remove(self.life)
        self.assertTrue(self.la.tick())
        self.assertTrue(os.path.exists(self.life))
        self.procs[0].code = 1                                # crashed
        self.assertTrue(self.la.tick())
        self.assertEqual(len(self.procs), 2)
        self.assertTrue(self.la.running())

    def test_gives_up_after_repeated_crashes(self):
        self.la.start()
        for _ in range(htl.MAX_RESTARTS):
            self.procs[-1].code = 1
            self.la.tick()
        self.assertEqual(len(self.procs), 1 + htl.MAX_RESTARTS)
        self.procs[-1].code = 1
        self.la.tick()
        self.assertTrue(self.la.gave_up())
        self.assertEqual(len(self.procs), 1 + htl.MAX_RESTARTS)   # no more spawns
        self.assertTrue(any("giving up" in m for m in self.logs))
        # switching tracking on again clears it
        self.assertTrue(self.la.start()[0])
        self.assertFalse(self.la.gave_up())

    def test_restart_budget_recovers_over_time(self):
        self.la.start()
        for _ in range(htl.MAX_RESTARTS):
            self.procs[-1].code = 1
            self.la.tick()
        self.clock[0] += htl.RESTART_WINDOW_S + 1
        self.procs[-1].code = 1
        self.la.tick()
        self.assertFalse(self.la.gave_up())

    def test_tick_after_stop_does_nothing(self):
        self.la.start()
        self.procs[0].code = 0
        self.la.stop()
        self.assertFalse(self.la.tick())
        self.assertEqual(len(self.procs), 1)

    def test_heartbeat_reads_the_service_status(self):
        self.assertIsNone(self.la.heartbeat())
        with open(self.alive, "w") as f:
            f.write('{"fps": 28.5, "hands": 2, "attached": true}')
        hb = self.la.heartbeat()
        self.assertEqual(hb["hands"], 2)
        self.assertLess(hb["age_s"], 5)
        with open(self.alive, "w") as f:
            f.write("garbage")
        self.assertIn("age_s", self.la.heartbeat())


if __name__ == "__main__":
    unittest.main()
