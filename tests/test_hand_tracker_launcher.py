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


class FakePump:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")
        return True

    def stop(self):
        self.calls.append("stop")
        return True


class CameraAndPumpTests(unittest.TestCase):
    """2026-11-17: the tracker learns its camera kind + MediaPipe confidences
    from the command line, and the Kinect pump lives exactly as long as the
    tracker (live 11-17: the Kinect fed the tracker only once per face-loop
    pass, ~7-10 fps)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        self.procs = []
        self.pump = FakePump()
        p = mock.patch.multiple(config, HAND_TRACK_PYTHON="py", HAND_TRACK_MODEL="model",
                                HAND_TRACK_PORT=8793, create=True)
        p.start()
        self.addCleanup(p.stop)
        self.exist = {"py", "model", htl.SERVICE_SCRIPT}

        def popen(argv, **kw):
            pr = FakeProc(argv, **kw)
            self.procs.append(pr)
            return pr
        self.la = htl.HandTrackerLauncher(
            popen=popen, clock=lambda: 1000.0, pid=777, log=lambda m: None,
            exists=lambda q: q in self.exist,
            alive_file=os.path.join(d, ".handtrack_alive"),
            jarvis_file=os.path.join(d, ".handtrack_jarvis_alive"),
            log_path=os.path.join(d, "ht.log"), start_keeper=False,
            kinect_pump=self.pump)

    def tearDown(self):
        self.dir.cleanup()

    def argv_for(self, **cfg):
        with mock.patch.multiple(config, create=True, **cfg):
            self.procs.clear()
            self.la._spawn()
        return self.procs[-1].argv

    def test_build_argv_only_adds_what_it_is_given(self):
        base = htl.build_argv("py", "s.py", "m", 1, 2, "a", "j")
        self.assertNotIn("--camera", base)
        self.assertNotIn("--det-conf", base)
        full = htl.build_argv("py", "s.py", "m", 1, 2, "a", "j", camera="kinect",
                              det_conf=0.4, presence_conf=0.45, track_conf=0.3)
        self.assertEqual(full[full.index("--camera") + 1], "kinect")
        self.assertEqual(full[full.index("--det-conf") + 1], "0.40")
        self.assertEqual(full[full.index("--presence-conf") + 1], "0.45")
        self.assertEqual(full[full.index("--track-conf") + 1], "0.30")
        self.assertNotIn("--camera", htl.build_argv("py", "s", "m", 1, 2, "a", "j",
                                                    camera="bogus"))

    def test_the_camera_kind_follows_the_setting(self):
        with mock.patch("core.handtrack_depth.holo_session", return_value=False):
            argv = self.argv_for(HAND_TRACK_CAMERA="kinect")
            self.assertEqual(argv[argv.index("--camera") + 1], "kinect")
            argv = self.argv_for(HAND_TRACK_CAMERA="primary")
            self.assertEqual(argv[argv.index("--camera") + 1], "webcam")
        with mock.patch("core.handtrack_depth.holo_session", return_value=True):
            argv = self.argv_for(HAND_TRACK_CAMERA="primary")         # a Kinect hologram
            self.assertEqual(argv[argv.index("--camera") + 1], "kinect")

    def test_the_confidences_come_from_config_clamped(self):
        argv = self.argv_for(HAND_TRACK_DET_CONF=0.4, HAND_TRACK_PRESENCE_CONF=2.0,
                             HAND_TRACK_TRACKING_CONF="junk")
        self.assertEqual(argv[argv.index("--det-conf") + 1], "0.40")
        self.assertEqual(argv[argv.index("--presence-conf") + 1], "0.95")   # clamped
        self.assertEqual(argv[argv.index("--track-conf") + 1], "0.50")      # bad -> default

    def test_the_shipped_defaults_are_mediapipes_own(self):
        for name in ("HAND_TRACK_DET_CONF", "HAND_TRACK_PRESENCE_CONF",
                     "HAND_TRACK_TRACKING_CONF"):
            self.assertIsInstance(getattr(config, name), float, name)
        with open(os.path.join(_ROOT, "core", "config.py"), encoding="utf-8") as f:
            src = f.read()
        for name in ("HAND_TRACK_DET_CONF", "HAND_TRACK_PRESENCE_CONF",
                     "HAND_TRACK_TRACKING_CONF"):
            self.assertRegex(src, rf"\n{name} = 0\.5\n", name)

    def test_a_saved_confidence_reaches_the_appended_keys(self):
        # The block sits after config's boot-time _apply_user_settings(), so it
        # reads its own three keys (fake file: never the real settings).
        saved = '{"HAND_TRACK_DET_CONF": 0.4, "HAND_TRACK_TRACKING_CONF": 7, ' \
                '"HAND_TRACK_PRESENCE_CONF": "x", "OTHER": 1}'
        keep = {k: getattr(config, k) for k in config._HAND_TRACK_CONF_KEYS}
        self.addCleanup(lambda: [setattr(config, k, v) for k, v in keep.items()])
        with mock.patch.object(config.os.path, "exists", return_value=True), \
                mock.patch("builtins.open", mock.mock_open(read_data=saved)):
            config._apply_hand_track_conf_settings()
        self.assertEqual(config.HAND_TRACK_DET_CONF, 0.4)
        self.assertEqual(config.HAND_TRACK_TRACKING_CONF, keep["HAND_TRACK_TRACKING_CONF"])
        self.assertEqual(config.HAND_TRACK_PRESENCE_CONF, keep["HAND_TRACK_PRESENCE_CONF"])
        with mock.patch.object(config.os.path, "exists", side_effect=OSError("x")):
            config._apply_hand_track_conf_settings()        # never raises

    def test_the_kinect_pump_lives_exactly_as_long_as_the_tracker(self):
        self.assertTrue(self.la.start()[0])
        self.assertEqual(self.pump.calls, ["start"])
        self.la.start()                                   # already running: re-asserts
        self.assertEqual(self.pump.calls, ["start", "start"])
        self.procs[0].code = 0
        self.la.stop()
        self.assertEqual(self.pump.calls[-1], "stop")

    def test_no_pump_when_the_tracker_cannot_start(self):
        self.exist.discard("model")
        self.assertFalse(self.la.start()[0])
        self.assertEqual(self.pump.calls, [])

    def test_a_broken_pump_never_breaks_the_launcher(self):
        class Boom:
            def start(self):
                raise RuntimeError("x")
            stop = start
        self.la._kinect_pump = Boom()
        self.assertTrue(self.la.start()[0])
        self.procs[0].code = 0
        self.la.stop()                                    # no raise

    def test_the_module_launcher_drives_the_real_pump_hook(self):
        self.assertIsInstance(htl.LAUNCHER._kinect_pump, htl._DepthPumpHook)
        with mock.patch("core.handtrack_depth.start_pump", return_value=True) as s, \
                mock.patch("core.handtrack_depth.stop_pump", return_value=True) as t:
            self.assertTrue(htl._DepthPumpHook.start())
            self.assertTrue(htl._DepthPumpHook.stop())
        self.assertEqual((s.call_count, t.call_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
