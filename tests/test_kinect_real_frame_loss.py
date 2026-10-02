"""The Kinect's "dies on open" verdict, only on REAL frame loss (2026-10-02).

WHAT WAS MEASURED (frames counted from the runtime's own arrival stamps; no
image content read). The Kinect v2 on the owner's desk drops off USB (Windows
Kernel-PnP 1010 "surprise removed") about 7.5 s after ANY process switches the
sensor on - colour+depth+body, colour only, or IKinectSensor::Open() with no
reader at all - on a direct root port as well as behind a hub, and never while
it is closed. JARVIS's verdict ("its stream died ... check its power supply")
was therefore right that frames were lost, but:

  * it measured the stream's life from the RESET, 4 s after the last frame
    ("died within 10.3s" for a stream that ran 6.3 s);
  * it said "drops off USB the moment it starts streaming" whatever it had
    seen - the phrase that sent the owner moving USB ports;
  * the reset nulled the runtime, talked to the gate and only THEN closed the
    old handle. Every PyKinectRuntime binds the ONE default sensor and close()
    calls IKinectSensor::Close() on it - not reference-counted: two runtimes
    held the same sensor pointer and closing the first stopped the second's
    frames at once (30/30/30 frames before, 0/0/0 after). So a poller that
    opened a runtime inside that window had it killed by JARVIS's own close:
    a JARVIS-made "died on open";
  * nothing confirmed the loss against the runtime's own stamps, and a failed
    open that touched the sensor was never logged.

Everything here is fake: pykinect2 is replaced by a module whose runtime binds
one shared fake sensor (modelling the measured Close() semantics), the clocks
are pinned, the gate is the real core/camera_gate.CameraGate on a frozen
clock, and the body pump is never started. Every test fails on what the old
code DOES (new names are reached through getattr), not on a missing name.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from audio import kinect_bridge as kb
from core import camera_gate as cg
from core import camera_tiles as ct

KINECT = "kinect"
BRIDGE = "kinect-bridge"
PERF0 = 50_000.0              # the pinned perf_counter (pykinect2's clock)

SAID_OFF_BUS = ("The Kinect drops off USB a few seconds after every start, "
                "sir. That's a hardware fault - check its power supply "
                "first. I'll only retry it every thirty minutes.")
SAID_CONNECTED = ("The Kinect's stream keeps dying a few seconds after every "
                  "start, sir, though it stays connected. I'll only retry it "
                  "every thirty minutes.")


class _Sensor:
    """The ONE process-wide IKinectSensor every runtime binds."""

    def __init__(self):
        self.is_open = False
        self.IsAvailable = True
        self.opens = 0

    def Open(self):
        self.is_open = True
        self.opens += 1

    def Close(self):
        self.is_open = False


class _Runtime:
    """PyKinectRuntime as measured: __init__ Opens the shared sensor and
    stamps every stream with pykinect2's clock; close() Closes the shared
    sensor (for every runtime on it) and drops its own reference."""

    def __init__(self, sensor, perf):
        self.sensor_ref = sensor
        self._sensor = sensor
        sensor.Open()
        self.closed = False
        t = perf()
        self._last_color_frame_time = t
        self._last_body_frame_time = t
        self._last_depth_frame_time = t

    def has_new_color_frame(self):
        return True

    def has_new_body_frame(self):
        return True

    def get_last_color_frame(self):
        return None

    def get_last_body_frame(self):
        return None

    def close(self):
        if self._sensor is not None:
            self._sensor.Close()
            self._sensor = None
        self.closed = True


class _Clock:
    def __init__(self, t: float = 5000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _kw(fn, **kw):
    """Only the keywords ``fn`` takes, so the old gate runs these tests."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kw.items() if k in params}


_CELLS = ("_runtime", "_open_error", "_negative_until", "_gate_hold_until",
          "_open_gate", "_service_down", "_service_said",
          "_last_body_frame_at", "_last_color_frame_at", "_last_depth_frame_at",
          "_color_time_seen", "_body_time_seen", "_depth_time_seen",
          "_link_lost_at", "_link_samples", "_link_sampled_at", "_lag_said_for")


class _BridgeBase(unittest.TestCase):

    def setUp(self):
        saved = {n: list(getattr(kb, n)) for n in _CELLS if hasattr(kb, n)}
        saved_enabled = kb._ENABLED
        fail_said = getattr(kb, "_open_fail_said", None)
        fail_saved = dict(fail_said) if isinstance(fail_said, dict) else None

        def _restore():
            for n, v in saved.items():
                getattr(kb, n)[:] = v
            kb._ENABLED = saved_enabled
            if fail_saved is not None:
                fail_said.clear()
                fail_said.update(fail_saved)
        self.addCleanup(_restore)
        if isinstance(fail_said, dict):
            fail_said.clear()
        self.perf = [PERF0]
        self.sensor = _Sensor()
        self.runtimes: list = []
        # For each construction: were ALL earlier runtimes already closed?
        self.built_on_closed: list = []
        pk2 = types.ModuleType("pykinect2.PyKinectV2")
        pk2.FrameSourceTypes_Color = 1
        pk2.FrameSourceTypes_Infrared = 2
        pk2.FrameSourceTypes_Depth = 8
        pk2.FrameSourceTypes_Body = 32
        rt_mod = types.ModuleType("pykinect2.PyKinectRuntime")

        def _ctor(_flags):
            self.built_on_closed.append(all(r.closed for r in self.runtimes))
            rt = _Runtime(self.sensor, lambda: self.perf[0])
            self.runtimes.append(rt)
            return rt
        rt_mod.PyKinectRuntime = _ctor
        kb._ENABLED = True
        kb._runtime[0] = None
        kb._open_error[0] = None
        kb._negative_until[0] = 0.0
        kb._gate_hold_until[0] = 0.0
        kb._service_down[0] = False
        kb._service_said[0] = False
        kb._open_gate[0] = None
        for p in (mock.patch.object(kb, "import_pykinect2",
                                    lambda: (pk2, rt_mod)),
                  mock.patch.object(kb, "_runtime_streams",
                                    lambda *a, **k: True),
                  mock.patch.object(kb, "_query_kinect_service_state",
                                    lambda: "running"),
                  mock.patch.object(kb, "start_body_pump", lambda: False),
                  mock.patch.object(kb, "_perf_now", lambda: self.perf[0],
                                    create=True)):
            p.start()
            self.addCleanup(p.stop)

    # ── helpers ───────────────────────────────────────────────────────────
    def _open(self):
        rt, err = kb.get_runtime()
        self.assertIsNotNone(rt, err)
        return rt

    def _stall(self, rt, quiet_s: float = 4.5):
        """No colour or body frame for ``quiet_s``: the bridge's clocks AND
        the runtime's own arrival stamps. Returns the monotonic 'now' at
        which the reset should look."""
        last = 1000.0
        kb._last_body_frame_at[0] = last
        kb._last_color_frame_at[0] = last
        rt._last_color_frame_time = self.perf[0] - quiet_s
        rt._last_body_frame_time = self.perf[0] - quiet_s
        return last + quiet_s

    def _watch(self, at: float) -> None:
        fn = getattr(kb, "_watch_sensor_link", None)
        if callable(fn):
            fn(now=at)

    def _reset(self, now: float) -> tuple:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            did = kb.reset_if_body_stale(now=now)
        return did, buf.getvalue()


class ReopenRaceTests(_BridgeBase):
    """The old handle is closed before anything can open a new runtime."""

    def test_a_poller_asking_mid_reset_never_opens_on_the_closing_sensor(self):
        old = self._open()
        now = self._stall(old)
        polled: list = []
        real_gate_call = kb._gate_call

        def _hook(method, *a, **k):
            # A 30 Hz poller (air-mouse / two-hand available(), the preview's
            # get_color_bgr) lands right after the reset dropped the runtime.
            if method == "unhold" and not polled:
                th = threading.Thread(target=lambda: polled.append(
                    kb.available()), daemon=True)
                th.start()
                th.join(5.0)
            return real_gate_call(method, *a, **k)
        with mock.patch.object(kb, "_gate_call", side_effect=_hook):
            did, _out = self._reset(now)
        self.assertTrue(did)
        self.assertTrue(old.closed)
        self.assertTrue(polled, "the poller never ran")
        self.assertTrue(all(self.built_on_closed),
                        "a new runtime was built while the old one still "
                        "held the shared sensor - its close() then killed "
                        "the new stream")
        rt, err = kb.get_runtime()
        self.assertIsNotNone(rt, err)
        self.assertFalse(rt.closed)
        self.assertTrue(self.sensor.is_open,
                        "the live runtime sits on a sensor the reset closed")

    def test_the_poller_is_told_and_not_put_on_a_cooldown(self):
        old = self._open()
        now = self._stall(old)
        polled: list = []
        real_gate_call = kb._gate_call

        def _hook(method, *a, **k):
            if method == "unhold" and not polled:
                th = threading.Thread(target=lambda: polled.append(
                    kb.get_runtime()), daemon=True)
                th.start()
                th.join(5.0)
            return real_gate_call(method, *a, **k)
        with mock.patch.object(kb, "_gate_call", side_effect=_hook):
            self._reset(now)
        self.assertEqual(polled[0][0], None,
                         "the poller opened a runtime mid-reset")
        self.assertEqual(polled[0][1], getattr(kb, "_RESET_IN_PROGRESS", "?"))
        self.assertLessEqual(kb._negative_until[0], time.monotonic(),
                             "a reset-in-progress answer earned a cooldown")
        self.assertIsNone(kb._open_error[0])
        self.assertIsNotNone(kb.get_runtime()[0], "no reopen after the reset")


class RealFrameLossTests(_BridgeBase):

    def test_frames_still_arriving_is_not_a_drop(self):
        """The bridge's own clocks stale, the runtime's stamps fresh: frames
        ARE arriving, so nothing is reset and the gate hears no drop."""
        clk = _Clock()
        gate = cg.CameraGate(clock=clk, log=lambda _l: None,
                             announce=lambda _m: None)
        kb.set_open_gate(gate)
        rt = self._open()
        kb._last_body_frame_at[0] = 1000.0
        kb._last_color_frame_at[0] = 1000.0
        rt._last_color_frame_time = self.perf[0] - 0.05
        rt._last_body_frame_time = self.perf[0] - 0.05
        with mock.patch.object(gate, "note_drop",
                               wraps=gate.note_drop) as drop:
            did, out = self._reset(1000.0 + kb.BODY_STALE_RESET_SEC + 1.0)
        self.assertFalse(did, "a runtime that is still delivering was reset")
        self.assertIs(kb._runtime[0], rt)
        self.assertFalse(rt.closed)
        self.assertEqual(drop.call_count, 0)
        self.assertIn("still delivering", out)
        # Said once per runtime, not 30 times a second.
        _did, out2 = self._reset(1000.0 + kb.BODY_STALE_RESET_SEC + 1.1)
        self.assertEqual(out2, "")

    def test_one_live_plane_is_enough(self):
        rt = self._open()
        now = self._stall(rt)
        rt._last_body_frame_time = self.perf[0] - 0.1     # body still arriving
        did, _out = self._reset(now)
        self.assertFalse(did)
        self.assertFalse(rt.closed)

    def test_a_real_stall_still_resets(self):
        rt = self._open()
        did, out = self._reset(self._stall(rt))
        self.assertTrue(did)
        self.assertTrue(rt.closed)
        self.assertIsNone(kb._runtime[0])
        self.assertIn("body AND color streams stale", out)


class HonestVerdictTests(_BridgeBase):
    """Bridge + the real gate: the verdict carries the stream's real life and
    says what the SDK saw."""

    def setUp(self):
        super().setUp()
        self.clk = _Clock()
        self.logs: list = []
        self.spoken: list = []
        self.gate = cg.CameraGate(clock=self.clk, log=self.logs.append,
                                  announce=self.spoken.append)
        kb.set_open_gate(self.gate)

    def _die(self, *, ran_s: float, available: "bool | None"):
        """Open when the gate allows; the stream runs ``ran_s``, then stops;
        the reset looks 4.5 s after the last frame. ``available`` is what the
        sensor says while the stream is quiet (None: nothing sampled)."""
        for _ in range(200):
            kb._gate_hold_until[0] = 0.0          # the bridge's re-ask
            rt, _err = kb.get_runtime()
            if rt is not None:
                break
            self.clk.advance(30.0)
        self.assertIsNotNone(rt)
        self.sensor.IsAvailable = True
        self.clk.advance(ran_s + 4.5)
        now = self._stall(rt, 4.5)
        if available is not None:
            self.sensor.IsAvailable = available
            for at in (0.5, 1.0, 1.5, 2.0):
                self._watch(now - 4.5 + at)
        did, out = self._reset(now)
        self.assertTrue(did)
        return out

    def test_off_the_bus_says_so_with_the_real_stream_life(self):
        outs = [self._die(ran_s=6.3, available=False) for _ in range(3)]
        self.assertIn("the sensor dropped off USB", outs[0])
        verdict = [ln for ln in self.logs if "opens in a row" in ln]
        self.assertEqual(len(verdict), 1, self.logs)
        self.assertIn("its stream died 6.3s after it opened", verdict[0])
        self.assertIn("drops off USB a few seconds after every start",
                      verdict[0])
        self.assertEqual(self.spoken, [SAID_OFF_BUS])
        self.assertIs(self.gate.snapshot()["devices"][KINECT]
                      .get("slow_retry_off_bus"), True)

    def test_a_stream_that_stops_while_connected_never_blames_power(self):
        outs = [self._die(ran_s=6.3, available=True) for _ in range(3)]
        self.assertIn("the sensor stayed connected", outs[0])
        verdict = [ln for ln in self.logs if "opens in a row" in ln]
        self.assertEqual(len(verdict), 1, self.logs)
        self.assertIn("not a USB or power drop", verdict[0])
        self.assertNotIn("check its power supply", verdict[0])
        self.assertEqual(self.spoken, [SAID_CONNECTED])
        self.assertNotIn("power", self.spoken[0])

    def test_the_safety_hold_is_unchanged(self):
        for _ in range(3):
            self._die(ran_s=6.3, available=False)
        self.clk.advance(120.0)
        kb._gate_hold_until[0] = 0.0
        rt, err = kb.get_runtime()
        self.assertIsNone(rt, "the sensor was reopened inside the slow retry")
        self.assertIn("backoff", err)
        self.assertEqual(len(self.runtimes), 3)


class FailedOpenIsLoggedTests(_BridgeBase):

    def test_a_gauntlet_that_got_no_frames_is_logged_once(self):
        buf = io.StringIO()
        with mock.patch.object(kb, "_runtime_streams", lambda *a, **k: False), \
             mock.patch.object(kb, "_OPEN_STREAM_RETRIES", 1), \
             mock.patch.object(kb, "_OPEN_STREAM_RETRY_SEC", 0.0), \
             contextlib.redirect_stdout(buf):
            for _ in range(3):
                kb._negative_until[0] = 0.0       # let it try again
                rt, err = kb.get_runtime()
                self.assertIsNone(rt)
        self.assertEqual(buf.getvalue().count("[kinect] open failed"), 1,
                         buf.getvalue())
        self.assertIn("streamed no frames", buf.getvalue())
        self.assertEqual(len(self.runtimes), 3)

    def test_a_refusal_that_never_touched_the_sensor_is_not(self):
        buf = io.StringIO()
        with mock.patch.object(kb, "_query_kinect_service_state",
                               lambda: "stopped"), \
             contextlib.redirect_stdout(buf):
            kb.get_runtime()
        self.assertNotIn("open failed", buf.getvalue())


class GateWordingTests(unittest.TestCase):
    """core/camera_gate.py on its own: the verdict is worded by the drop's
    evidence, and the onset can be given as 'seconds ago'."""

    def _run(self, key=KINECT, comp=BRIDGE, gone_at_drop=False, **drop_kw):
        clk = _Clock(1000.0)
        logs: list = []
        spoken: list = []
        gone = [False]
        g = cg.CameraGate(clock=clk, log=logs.append, announce=spoken.append,
                          presence=((lambda _k: not gone[0]) if gone_at_drop
                                    else None),
                          labeler=lambda k: {KINECT: "the Kinect"}.get(k, ""))
        for _ in range(cg.DIES_ON_OPEN_COUNT):
            for _i in range(200):
                d = g.begin(key, comp)
                if d.allowed:
                    break
                clk.advance(min(max(0.5, d.wait_s), 60.0))
            self.assertTrue(d.allowed, d)
            g.end(key, comp, True)
            g.hold(key, comp)
            clk.advance(10.3)
            g.unhold(key, comp)
            gone[0] = gone_at_drop          # off the device list right now
            g.note_drop(key, comp, **_kw(g.note_drop, **drop_kw))
            gone[0] = False                 # ...and back (it re-enumerates)
        return g, logs, spoken

    def test_off_bus_evidence(self):
        _g, logs, spoken = self._run(off_bus=True)
        self.assertEqual(spoken, [SAID_OFF_BUS])

    def test_stayed_connected_evidence(self):
        _g, logs, spoken = self._run(off_bus=False)
        self.assertEqual(spoken, [SAID_CONNECTED])
        self.assertIn("not a USB or power drop", logs[-1])

    def test_a_device_gone_from_the_list_is_off_the_bus(self):
        _g, _l, spoken = self._run(gone_at_drop=True)
        self.assertEqual(spoken, [SAID_OFF_BUS])

    def test_no_evidence_hedges(self):
        _g, logs, spoken = self._run()
        self.assertIn("If it's dropping off USB, check its power supply",
                      spoken[0])
        self.assertNotIn("the moment it starts streaming", spoken[0])

    def test_onset_ago_measures_the_stream_not_the_report(self):
        _g, logs, _s = self._run(onset_ago=4.0, off_bus=True)
        verdict = [ln for ln in logs if "opens in a row" in ln]
        self.assertIn("its stream died 6.3s after it opened", verdict[0])

    def test_the_evidence_survives_a_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "doo.json")
        clk = _Clock(1000.0)
        g = cg.CameraGate(clock=clk, log=lambda _l: None,
                          announce=lambda _m: None, doo_state_path=path)
        for _ in range(cg.DIES_ON_OPEN_COUNT):
            for _i in range(200):
                d = g.begin(KINECT, BRIDGE)
                if d.allowed:
                    break
                clk.advance(min(max(0.5, d.wait_s), 60.0))
            g.end(KINECT, BRIDGE, True)
            clk.advance(6.0)
            g.note_drop(KINECT, BRIDGE, **_kw(g.note_drop, off_bus=True))
        with open(path, encoding="utf-8") as fh:
            self.assertIs(json.load(fh)["devices"][KINECT].get("off_bus"),
                          True)
        g2 = cg.CameraGate(clock=clk, log=lambda _l: None,
                           announce=lambda _m: None, doo_state_path=path)
        self.assertIs(g2.snapshot()["devices"][KINECT]
                      .get("slow_retry_off_bus"), True)


class TileWordingTests(unittest.TestCase):
    """The dashboard tile for a device on the slow retry."""

    def _msg(self, off_bus):
        dev = {"hold_s": 1500.0, "slow_retry_s": 1800.0}
        if off_bus is not ...:
            dev["slow_retry_off_bus"] = off_bus
        return ct.gate_summary({"devices": {KINECT: dev}}, KINECT)["message"]

    def test_off_bus(self):
        m = self._msg(True)
        self.assertIn("dropped off USB a few seconds after each start", m)
        self.assertIn("Check its power supply", m)

    def test_stayed_connected_does_not_mention_power(self):
        m = self._msg(False)
        self.assertIn("stayed connected", m)
        self.assertNotIn("power", m)

    def test_unknown_hedges(self):
        for v in (None, ...):
            m = self._msg(v)
            self.assertIn("If it is dropping off USB", m)
            self.assertNotIn("each time it started streaming", m)


if __name__ == "__main__":
    unittest.main()
