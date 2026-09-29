"""audio/kinect_bridge.py obeys the process-wide camera open gate (2026-09-29).

The Kinect sits behind the same chained USB hubs as the owner's webcams, and
its stale-stream reset reopened the sensor every BODY_STALE_RESET_SEC while
both streams were quiet - one of the independent retry loops behind the
2026-09-29 hub-reset storm. With the gate installed (the monolith does it at
import) every PyKinectRuntime() open asks it first; a USB-storm cool-down or a
backoff rung means the sensor is not touched at all.

Nothing here touches a sensor: pykinect2 is faked, the runtime is a plain
object, and the gate is the real core/camera_gate.CameraGate on a frozen
clock. The body pump is never started.
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

from audio import kinect_bridge as kb
from core import camera_gate as cg


class _Runtime:
    def __init__(self):
        self.closed = False

    def has_new_color_frame(self):
        return True

    def has_new_body_frame(self):
        return True

    def close(self):
        self.closed = True


class _Clock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t


class KinectOpensGoThroughTheGateTests(unittest.TestCase):

    def setUp(self):
        self.clock = _Clock()
        self.logs: list = []
        self.gate = cg.CameraGate(clock=self.clock, log=self.logs.append,
                                  announce=lambda _m: None)
        self.opens: list = []
        pk2 = types.ModuleType("pykinect2.PyKinectV2")
        pk2.FrameSourceTypes_Color = 1
        pk2.FrameSourceTypes_Infrared = 2
        pk2.FrameSourceTypes_Depth = 8
        pk2.FrameSourceTypes_Body = 32
        rt_mod = types.ModuleType("pykinect2.PyKinectRuntime")

        def _ctor(_flags):
            rt = _Runtime()
            self.opens.append(rt)
            return rt
        rt_mod.PyKinectRuntime = _ctor
        saved = (kb._ENABLED, kb._open_gate[0], kb._runtime[0],
                 kb._open_error[0], kb._negative_until[0],
                 kb._gate_hold_until[0], kb._service_down[0],
                 kb._service_said[0])

        def _restore():
            (kb._ENABLED, kb._open_gate[0], kb._runtime[0], kb._open_error[0],
             kb._negative_until[0], kb._gate_hold_until[0],
             kb._service_down[0], kb._service_said[0]) = saved
        self.addCleanup(_restore)
        kb._service_down[0] = False
        kb._service_said[0] = False
        # The runtime-service state is THIS machine's; tests say it themselves.
        self.service = ["running"]
        _svc = mock.patch.object(kb, "_query_kinect_service_state",
                                 side_effect=lambda: self.service[0])
        _svc.start()
        self.addCleanup(_svc.stop)
        kb._ENABLED = True
        kb._runtime[0] = None
        kb._open_error[0] = None
        kb._negative_until[0] = 0.0
        kb.set_open_gate(self.gate)
        for p in (mock.patch.object(kb, "import_pykinect2",
                                    lambda: (pk2, rt_mod)),
                  mock.patch.object(kb, "_runtime_streams",
                                    lambda *a, **k: True),
                  mock.patch.object(kb, "start_body_pump", lambda: False)):
            p.start()
            self.addCleanup(p.stop)

    def _storm(self):
        self.gate.note_drop("name:synth-a", "face-track")
        self.gate.note_drop("name:synth-b", "face-track")
        self.assertTrue(self.gate.storm_active())

    def test_a_storm_cool_down_opens_nothing(self):
        self._storm()
        rt, err = kb.get_runtime()
        self.assertIsNone(rt)
        self.assertEqual(self.opens, [], "the sensor was opened mid-storm")
        self.assertIn("usb-storm", err)
        ok, err2 = kb.available()
        self.assertFalse(ok)
        self.assertEqual(self.opens, [])
        self.assertIn("camera gate", err2)

    def test_a_refusal_is_remembered_so_the_pump_does_not_re_ask_every_tick(self):
        self._storm()
        with mock.patch.object(self.gate, "begin",
                               wraps=self.gate.begin) as begin:
            for _ in range(30):                  # one second of 30 Hz pump
                kb.get_runtime()
        self.assertEqual(begin.call_count, 1)

    def test_an_allowed_open_is_reported_and_held(self):
        rt, err = kb.get_runtime()
        self.assertIsNotNone(rt)
        self.assertIsNone(err)
        self.assertEqual(len(self.opens), 1)
        snap = self.gate.snapshot()["devices"]["kinect"]
        self.assertEqual(snap["held_by"], "kinect-bridge")
        self.assertEqual(snap["in_flight"], "")
        # The self-diagnostic's MSMF scan reaches the Kinect too: refused.
        self.assertEqual(self.gate.begin("kinect", "self-diag").reason, "held")

    def test_a_stale_reset_is_a_drop_and_the_reopen_is_a_recovery(self):
        kb.get_runtime()
        # Both streams stale: the bridge resets its runtime.
        kb._last_body_frame_at[0] = 1.0
        kb._last_color_frame_at[0] = 1.0
        self.assertTrue(kb.reset_if_body_stale(now=1.0 + kb.BODY_STALE_RESET_SEC + 1))
        snap = self.gate.snapshot()["devices"]["kinect"]
        self.assertEqual(snap["held_by"], "", "a reset runtime is still held")
        self.assertTrue(snap["recovering"])
        # First recovery reopen: immediate.
        rt, _ = kb.get_runtime()
        self.assertIsNotNone(rt)
        self.assertEqual(len(self.opens), 2)
        # It streams briefly, then goes stale again 5 s later: that second
        # recovery must NOT reopen at once - it waits out the rung the first
        # recovery spent (30 s).
        self.gate.note_frame("kinect")
        kb._last_body_frame_at[0] = 1.0
        kb._last_color_frame_at[0] = 1.0
        self.clock.t += 5.0
        self.assertTrue(kb.reset_if_body_stale(now=1.0 + kb.BODY_STALE_RESET_SEC + 1))
        rt, err = kb.get_runtime()
        self.assertIsNone(rt, "the sensor was reopened 5 s after a recovery")
        self.assertIn("backoff", err)
        self.assertEqual(len(self.opens), 2)

    def test_a_failed_open_counts_toward_the_breaker(self):
        # The Kinect plus one webcam failing = two devices; a third failure
        # within the minute trips the storm.
        with mock.patch.object(kb, "_runtime_streams", lambda *a, **k: False), \
             mock.patch.object(kb, "_OPEN_STREAM_RETRIES", 1), \
             mock.patch.object(kb, "_OPEN_STREAM_RETRY_SEC", 0.0):
            rt, _err = kb.get_runtime()
        self.assertIsNone(rt)
        d = self.gate.begin("name:synth-a", "face-track")
        self.gate.end("name:synth-a", "face-track", False)
        self.clock.t += 31.0
        d = self.gate.begin("name:synth-a", "face-track")
        self.assertTrue(d.allowed, d)
        self.gate.end("name:synth-a", "face-track", False)
        self.assertTrue(self.gate.storm_active())

    def test_close_releases_the_hold(self):
        kb.get_runtime()
        kb.close()
        self.assertEqual(self.gate.snapshot()["devices"]["kinect"]["held_by"], "")

    def test_a_stopped_runtime_service_is_said_once_and_not_retried(self):
        """KinectMonitor Stopped (the owner's desk since 2026-09-04): the
        bridge must not open the sensor, must say so ONCE, and must not
        re-ask every pump tick."""
        import contextlib
        import io
        self.service[0] = "stopped"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(5):
                rt, err = kb.get_runtime()
                self.assertIsNone(rt)
                self.assertIn("not running", err)
                kb._negative_until[0] = 0.0      # force a re-check each time
            self.assertTrue(kb.service_down())
        self.assertEqual(self.opens, [], "the sensor was opened with its "
                                         "runtime service stopped")
        self.assertEqual(buf.getvalue().count("is STOPPED"), 1, buf.getvalue())
        # Nothing was counted against the device: it was never touched.
        self.assertNotIn("kinect", self.gate.snapshot()["devices"])
        # The owner starts the service: the next ask opens, and says so once.
        self.service[0] = "running"
        with contextlib.redirect_stdout(buf):
            rt, err = kb.get_runtime()
        self.assertIsNotNone(rt)
        self.assertFalse(kb.service_down())
        self.assertEqual(buf.getvalue().count("opening the sensor again"), 1)

    def test_a_stopped_service_is_rechecked_at_its_own_cadence(self):
        self.service[0] = "stopped"
        kb.get_runtime()
        self.assertGreaterEqual(kb._negative_until[0] - kb.time.monotonic(),
                                kb._SERVICE_RECHECK_S - 1.0)

    def test_without_a_gate_nothing_changes(self):
        kb.set_open_gate(None)
        rt, err = kb.get_runtime()
        self.assertIsNotNone(rt)
        self.assertIsNone(err)
        self.assertEqual(len(self.opens), 1)


if __name__ == "__main__":
    unittest.main()
