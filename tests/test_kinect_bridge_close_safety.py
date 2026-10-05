"""audio/kinect_bridge - a runtime is never closed under a reader, frames leave
the bridge as the bridge's own memory, and a runtime's frame thread stops
before its close releases anything (v2.0.180).

WHY (2026-10-04): v2.0.179 died with "Fatal Python error: Aborted" while the
face-track watchdog showed the preview producer frozen at
``canvas = base_bgr.copy()`` over a Kinect frame. The crash dumps proved the
Kinect innocent (a CTranslate2 thread-exit abort; see tests/test_ct2_host.py)
- but the investigation found real hazards on this path, fixed here as
defence in depth:
  * on v2.0.178 the same day, "AttributeError: 'NoneType' object has no
    attribute 'GetFrameArrivedEventData'" tracebacks came from pykinect2's
    frame thread each time the bridge's stale-stream reset closed a runtime:
    PyKinectRuntime.close() drops its readers, closes its event handle and
    closes the sensor while its own (never-joined) frame thread is mid-frame;
  * close()/reset never waited for a reader that was inside the runtime (the
    coordinate-mapper COM calls, the frame getters), and a mapper kept past a
    reset kept calling into the closed runtime;
  * the frame arrays were safe only because pykinect2 0.1.0 happens to
    numpy.copy in its getters - a build that returns a view of its ctypes
    buffer would hand the preview memory the frame thread overwrites.

All with fakes: no Kinect, no pykinect2.

    python -m unittest tests.test_kinect_bridge_close_safety
"""
from __future__ import annotations

import _thread
import ctypes
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from audio import kinect_bridge as kb
from tests.test_kinect_bridge import (_BridgeBase, _FakeRuntime,
                                      _fake_pk2_module, _patch_loader)

_COLOR_N = 1920 * 1080 * 4
_DEPTH_N = 512 * 424


class _RecLock:
    """A lock that counts acquisitions (the runtime's frame lock)."""

    def __init__(self):
        self._l = threading.Lock()
        self.acquired = 0

    def acquire(self, blocking=True, timeout=-1):
        ok = self._l.acquire(blocking, timeout)
        if ok:
            self.acquired += 1
        return ok

    def release(self):
        self._l.release()


class OwnedFrameTests(_BridgeBase):
    def test_a_view_of_runtime_memory_is_copied_under_the_frame_lock(self):
        buf = (ctypes.c_ubyte * _COLOR_N)()
        view = np.ctypeslib.as_array(buf)          # the runtime's own buffer
        rt = _FakeRuntime(color=view)
        rt._color_frame_lock = _RecLock()
        _patch_loader(self, rt)
        frame = kb.get_color_bgr()
        self.assertIsNotNone(frame)
        self.assertFalse(np.shares_memory(frame, view))
        buf[0] = 77                                # the frame thread writes on
        self.assertEqual(int(frame[0, 0, 0]), 0)
        self.assertEqual(rt._color_frame_lock.acquired, 1)

    def test_an_owned_frame_is_not_copied_again(self):
        # pykinect2 0.1.0 already returns numpy.copy(...): zero extra cost.
        owned = np.zeros(_COLOR_N, dtype=np.uint8)
        rt = _FakeRuntime(color=owned)
        rt._color_frame_lock = _RecLock()
        _patch_loader(self, rt)
        frame = kb.get_color_bgr()
        self.assertTrue(np.shares_memory(frame, owned))
        self.assertEqual(rt._color_frame_lock.acquired, 0)

    def test_depth_view_is_copied(self):
        buf = (ctypes.c_ushort * _DEPTH_N)()
        view = np.ctypeslib.as_array(buf)
        rt = _FakeRuntime(depth=view)
        _patch_loader(self, rt)
        depth = kb.get_depth()
        self.assertIsNotNone(depth)
        self.assertFalse(np.shares_memory(depth, view))


class _SlowReaderRuntime(_FakeRuntime):
    """A colour read that takes a while (the getter's copy) and a log of the
    order things happen in."""

    def __init__(self):
        super().__init__(color=np.zeros(_COLOR_N, dtype=np.uint8))
        self.in_read = threading.Event()
        self.release_read = threading.Event()
        self.log: list = []

    def get_last_color_frame(self):
        self.in_read.set()
        self.release_read.wait(3.0)
        self.log.append("read-done")
        return super().get_last_color_frame()

    def close(self):
        self.log.append("close")
        super().close()


class NoCloseUnderAReaderTests(_BridgeBase):
    def test_close_waits_for_a_reader_inside_the_runtime(self):
        rt = _SlowReaderRuntime()
        _patch_loader(self, rt)
        self.assertIsNotNone(kb.get_runtime()[0])
        reader = threading.Thread(
            target=lambda: kb.get_color_bgr(require_new=False), daemon=True)
        reader.start()
        self.assertTrue(rt.in_read.wait(2.0))
        closer = threading.Thread(target=kb.close, daemon=True)
        closer.start()
        time.sleep(0.15)
        self.assertNotIn("close", rt.log)          # still waiting for the reader
        rt.release_read.set()
        reader.join(3.0)
        closer.join(3.0)
        self.assertEqual(rt.log, ["read-done", "close"])

    def test_close_does_not_wait_forever_on_a_stuck_reader(self):
        rt = _SlowReaderRuntime()
        _patch_loader(self, rt)
        kb.get_runtime()
        reader = threading.Thread(
            target=lambda: kb.get_color_bgr(require_new=False), daemon=True)
        reader.start()
        self.assertTrue(rt.in_read.wait(2.0))
        t0 = time.monotonic()
        with mock.patch.object(kb, "_CLOSE_DRAIN_S", 0.2):
            kb.close()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertIn("close", rt.log)             # bounded: closed anyway
        rt.release_read.set()
        reader.join(3.0)

    def test_a_mapper_kept_past_close_never_calls_into_the_closed_runtime(self):
        calls = []

        class _Pt:
            def __init__(self, x, y):
                self.x, self.y = x, y

        class _Mapper:
            def MapCameraPointToColorSpace(self, pt):
                calls.append((pt.x, pt.y, pt.z))
                return _Pt(pt.x, pt.y)

        class _CamPt:
            x = y = z = 0.0

        rt = _FakeRuntime()
        rt._mapper = _Mapper()
        pk2 = _fake_pk2_module()
        pk2.CameraSpacePoint = _CamPt
        rt_mod = types.ModuleType("pykinect2.PyKinectRuntime")
        rt_mod.PyKinectRuntime = lambda flags: rt
        with mock.patch.object(kb, "import_pykinect2", lambda: (pk2, rt_mod)):
            kb.get_runtime()
            mapper = kb.get_color_space_mapper()
            self.assertEqual(mapper(1.0, 2.0, 3.0), (1.0, 2.0))
            kb.close()
            self.assertTrue(rt.closed)
            self.assertIsNone(mapper(1.0, 2.0, 3.0))
        self.assertEqual(len(calls), 1)            # nothing after the close

    def test_reset_closes_through_the_same_drain(self):
        # The stale-stream reset releases the runtime through the ONE closer.
        rt = _FakeRuntime(color=np.zeros(_COLOR_N, dtype=np.uint8))
        _patch_loader(self, rt)
        kb.get_runtime()
        old = time.monotonic() - 60.0
        kb._last_body_frame_at[0] = old
        kb._last_color_frame_at[0] = old
        with mock.patch.object(kb, "_drain_readers",
                               wraps=kb._drain_readers) as drain:
            self.assertTrue(kb.reset_if_body_stale())
        self.assertTrue(rt.closed)
        drain.assert_called_once_with(rt)


def _pk_runtime_class():
    """A stand-in with pykinect2 0.1.0's close / frame-thread shape: close()
    sets the close event and, in the same breath, drops the frame readers and
    the sensor; the frame thread (started with _thread, never joined) handles
    a frame for a while and then touches its reader."""

    class PyKinectRuntime:
        def __init__(self, flags):
            self._close_event = threading.Event()   # the Win32 event
            self._color_frame_reader = types.SimpleNamespace(
                GetFrameArrivedEventData=lambda h: h)
            self._sensor = object()
            self.errors: list = []
            self.handling = threading.Event()
            self.copy_gate = threading.Event()   # the frame copy ends when set
            self.exited = threading.Event()
            _thread.start_new_thread(self.kinect_frame_thread, ())

        def kinect_frame_thread(self):
            try:
                while not self._close_event.wait(0.002):
                    self.handling.set()               # handle_color_arrived
                    self.copy_gate.wait(0.3)          # the frame copy
                    try:
                        self._color_frame_reader.GetFrameArrivedEventData(0)
                    except AttributeError as e:       # the 178 traceback
                        self.errors.append(e)
                        return
            finally:
                self.exited.set()

        def close(self):
            if self._sensor is not None:
                self._close_event.set()
                self._color_frame_reader = None
                self._sensor = None

    return PyKinectRuntime


def _set_event(rt):
    rt._close_event.set()
    return True


class FrameThreadGuardTests(unittest.TestCase):
    def _mod(self):
        mod = types.ModuleType("pykinect2.PyKinectRuntime")
        mod.PyKinectRuntime = _pk_runtime_class()
        return mod

    def test_unguarded_close_races_its_frame_thread(self):
        # Control: the fake reproduces the race the live log showed.
        rt = self._mod().PyKinectRuntime(0)
        self.assertTrue(rt.handling.wait(1.0))
        rt.close()                                # mid-frame, as live
        rt.copy_gate.set()
        self.assertTrue(rt.exited.wait(1.0))
        self.assertEqual(len(rt.errors), 1)
        self.assertIsInstance(rt.errors[0], AttributeError)

    def test_guarded_close_stops_the_frame_thread_first(self):
        mod = self._mod()
        self.assertTrue(kb._install_frame_thread_guard(mod))
        self.assertFalse(kb._install_frame_thread_guard(mod))   # idempotent
        with mock.patch.object(kb, "_signal_close_event", _set_event):
            rt = mod.PyKinectRuntime(0)
            self.assertTrue(rt.handling.wait(1.0))
            rt.close()
        self.assertTrue(rt._bridge_frame_done.is_set())   # stopped BEFORE close
        self.assertTrue(rt.exited.is_set())
        self.assertIsNone(rt._sensor)
        self.assertEqual(rt.errors, [])

    def test_frame_thread_stop_is_bounded(self):
        mod = self._mod()
        kb._install_frame_thread_guard(mod)
        rt = mod.PyKinectRuntime(0)
        self.assertTrue(rt.handling.wait(1.0))
        t0 = time.monotonic()
        # The close event cannot be signalled: the close goes ahead at once,
        # exactly as it did before the guard.
        with mock.patch.object(kb, "_signal_close_event", lambda rt: False):
            rt.close()
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertIsNone(rt._sensor)
        rt.exited.wait(1.0)

    def test_a_frame_thread_exception_is_one_line_not_a_dead_traceback(self):
        class _Raiser:
            def __init__(self, flags):
                self._sensor = object()
                self._close_event = 1
                _thread.start_new_thread(self.kinect_frame_thread, ())

            def kinect_frame_thread(self):
                raise AttributeError("'NoneType' object has no attribute "
                                     "'GetFrameArrivedEventData'")

            def close(self):
                self._sensor = None
        mod = types.ModuleType("pykinect2.PyKinectRuntime")
        mod.PyKinectRuntime = _Raiser
        kb._install_frame_thread_guard(mod)
        with mock.patch("builtins.print") as pr:
            rt = mod.PyKinectRuntime(0)
            self.assertTrue(rt._bridge_frame_done.wait(1.0))
        lines = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("frame thread stopped on AttributeError", lines)

    def test_import_pykinect2_installs_the_guard(self):
        pk2 = _fake_pk2_module()
        rt_mod = self._mod()
        with mock.patch.object(kb.importlib, "import_module", return_value=None), \
             mock.patch.object(kb, "_load_patched", return_value=None), \
             mock.patch.dict(kb.sys.modules,
                             {"pykinect2.PyKinectV2": pk2,
                              "pykinect2.PyKinectRuntime": rt_mod}):
            got_pk2, got_rt = kb.import_pykinect2()
        self.assertIs(got_rt, rt_mod)
        self.assertTrue(getattr(rt_mod.PyKinectRuntime.kinect_frame_thread,
                                "_bridge_guarded", False))
        self.assertTrue(getattr(rt_mod.PyKinectRuntime.close,
                                "_bridge_guarded", False))


if __name__ == "__main__":
    unittest.main()
