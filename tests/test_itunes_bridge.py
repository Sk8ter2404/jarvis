"""Tests for audio.itunes_bridge — the lazy iTunes COM client.

The whole point of this module is to NEVER touch win32com / pythoncom at import
time and to gate every COM access behind is_running() / force / auto-launch
checks. These tests exercise that gating logic without a real iTunes, real COM,
or even pywin32 being importable:

  * win32com.client / pythoncom / psutil are injected as fakes into sys.modules
    for the duration of each test (the module imports them lazily INSIDE
    get_client / is_running, so the fakes are picked up). This makes the COM
    branches run identically on Windows AND on the bare-Linux CI runner where
    pywin32 is a Windows-only, absent module.
  * subprocess.Popen (launch), time.sleep, and time.time are patched so no
    process spawns and the ready-polling loops resolve instantly.

The module must still IMPORT on CI — verified by the import at module top
(it does no top-level win32com/pythoncom/psutil import). stdlib unittest + mock.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from audio import itunes_bridge as ib


# ─────────────────────────────────────────────────────────────────────────
# fake-module factories (injected into sys.modules so the lazy imports inside
# get_client/is_running pick them up on any OS)
# ─────────────────────────────────────────────────────────────────────────
def _fake_psutil(names):
    """A psutil stand-in whose process_iter yields procs with the given
    names (list of strings)."""
    mod = types.ModuleType("psutil")

    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    class _Proc:
        def __init__(self, name):
            self.info = {"name": name}

    def process_iter(_attrs):
        return [_Proc(n) for n in names]

    mod.NoSuchProcess = NoSuchProcess
    mod.AccessDenied = AccessDenied
    mod.process_iter = process_iter
    return mod


def _fake_com(app=None, get_active_exc=None, dispatch_exc=None,
              dispatch_seq=None):
    """Build fake win32com (with .client) and pythoncom modules.

    app           — the object GetActiveObject/Dispatch return on success.
    get_active_exc— exception GetActiveObject raises (None = succeeds).
    dispatch_exc  — exception Dispatch raises every call (None = succeeds).
    dispatch_seq  — optional list; each Dispatch call pops the next item, and
                    if it's an Exception instance it's raised, else returned.
    """
    win32com = types.ModuleType("win32com")
    client = types.ModuleType("win32com.client")

    def get_active(progid):
        if get_active_exc is not None:
            raise get_active_exc
        return app

    seq = list(dispatch_seq) if dispatch_seq is not None else None

    def dispatch(progid):
        if seq is not None:
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if dispatch_exc is not None:
            raise dispatch_exc
        return app

    client.GetActiveObject = get_active
    client.Dispatch = dispatch
    win32com.client = client

    pythoncom = types.ModuleType("pythoncom")
    pythoncom.CoInitialize = mock.MagicMock()
    return win32com, client, pythoncom


def _ready_app():
    """A fake iTunes app whose LibraryPlaylist.Tracks.Count is accessible —
    i.e. the library is 'ready' so the wait_for_ready poll returns immediately."""
    app = mock.MagicMock()
    app.LibraryPlaylist.Tracks.Count = 1234
    return app


class _BridgeBase(unittest.TestCase):
    """Reset bridge module globals + the per-thread COM TLS between tests so
    state never leaks (e.g. a previous _ensure_com_init flag)."""

    def setUp(self):
        self._orig_auto = ib._AUTO_LAUNCH
        self._orig_check = ib._apple_music_active_check
        self.addCleanup(self._restore)
        # Fresh TLS so _ensure_com_init runs CoInitialize again.
        ib._itunes_com_tls = ib.threading.local()

    def _restore(self):
        ib._AUTO_LAUNCH = self._orig_auto
        ib._apple_music_active_check = self._orig_check

    def _inject(self, name, module):
        old = sys.modules.get(name)
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

        def restore():
            if old is not None:
                sys.modules[name] = old
            else:
                sys.modules.pop(name, None)
        self.addCleanup(restore)

    def _inject_com(self, win32com, client, pythoncom):
        self._inject("win32com", win32com)
        self._inject("win32com.client", client)
        self._inject("pythoncom", pythoncom)


# ─────────────────────────────────────────────────────────────────────────
# configuration hooks
# ─────────────────────────────────────────────────────────────────────────
class ConfigHookTests(_BridgeBase):
    def test_set_get_auto_launch(self):
        ib.set_auto_launch(True)
        self.assertTrue(ib.get_auto_launch())
        ib.set_auto_launch(False)
        self.assertFalse(ib.get_auto_launch())

    def test_set_auto_launch_coerces_to_bool(self):
        ib.set_auto_launch("yes")     # truthy → True
        self.assertIs(ib.get_auto_launch(), True)
        ib.set_auto_launch(0)
        self.assertIs(ib.get_auto_launch(), False)

    def test_set_apple_music_active_check(self):
        fn = lambda: True
        ib.set_apple_music_active_check(fn)
        self.assertIs(ib._apple_music_active_check, fn)
        ib.set_apple_music_active_check(None)
        self.assertIsNone(ib._apple_music_active_check)


# ─────────────────────────────────────────────────────────────────────────
# find_itunes_exe / is_running
# ─────────────────────────────────────────────────────────────────────────
class FindExeTests(_BridgeBase):
    def test_returns_first_existing_candidate(self):
        target = ib._ITUNES_EXE_CANDIDATES[0]
        with mock.patch.object(ib.os.path, "isfile",
                               side_effect=lambda p: p == target):
            self.assertEqual(ib.find_itunes_exe(), target)

    def test_returns_none_when_absent(self):
        with mock.patch.object(ib.os.path, "isfile", return_value=False):
            self.assertIsNone(ib.find_itunes_exe())


class IsRunningTests(_BridgeBase):
    def test_true_when_itunes_in_process_list(self):
        self._inject("psutil", _fake_psutil(["chrome.exe", "iTunes.exe"]))
        self.assertTrue(ib.is_running())

    def test_false_when_not_in_list(self):
        self._inject("psutil", _fake_psutil(["chrome.exe", "explorer.exe"]))
        self.assertFalse(ib.is_running())

    def test_case_insensitive_match(self):
        self._inject("psutil", _fake_psutil(["ITUNES.EXE"]))
        self.assertTrue(ib.is_running())

    def test_none_name_tolerated(self):
        self._inject("psutil", _fake_psutil([None, "iTunes.exe"]))
        self.assertTrue(ib.is_running())

    def test_psutil_absent_returns_false(self):
        # psutil = None in sys.modules → `import psutil` raises ImportError.
        self._inject("psutil", None)
        # Guard: if the real psutil is importable, this still works because the
        # injected None shadows it for the import inside is_running.
        with mock.patch.dict(sys.modules, {"psutil": None}):
            self.assertFalse(ib.is_running())

    def test_proc_access_error_skipped(self):
        # A proc whose .info access raises NoSuchProcess is skipped, not fatal.
        pmod = _fake_psutil(["iTunes.exe"])

        class _BadProc:
            @property
            def info(self):
                raise pmod.NoSuchProcess()

        good = list(pmod.process_iter(None))
        pmod.process_iter = lambda _a: [_BadProc()] + good
        self._inject("psutil", pmod)
        self.assertTrue(ib.is_running())   # the good iTunes proc still matches


# ─────────────────────────────────────────────────────────────────────────
# launch
# ─────────────────────────────────────────────────────────────────────────
class LaunchTests(_BridgeBase):
    def test_exe_not_found(self):
        with mock.patch.object(ib, "find_itunes_exe", return_value=None):
            ok, err = ib.launch()
        self.assertFalse(ok)
        self.assertIn("not found", err)

    def test_launches_detached(self):
        with mock.patch.object(ib, "find_itunes_exe",
                               return_value=r"C:\iTunes\iTunes.exe"), \
                mock.patch.object(ib.subprocess, "Popen") as mpopen:
            ok, err = ib.launch()
        self.assertTrue(ok)
        self.assertIsNone(err)
        mpopen.assert_called_once()
        # exe path is the sole positional arg list element
        self.assertEqual(mpopen.call_args[0][0], [r"C:\iTunes\iTunes.exe"])

    def test_popen_failure_reported(self):
        with mock.patch.object(ib, "find_itunes_exe",
                               return_value=r"C:\iTunes\iTunes.exe"), \
                mock.patch.object(ib.subprocess, "Popen",
                                  side_effect=OSError("spawn denied")):
            ok, err = ib.launch()
        self.assertFalse(ok)
        self.assertIn("failed to launch iTunes", err)
        self.assertIn("spawn denied", err)


# ─────────────────────────────────────────────────────────────────────────
# _ensure_com_init (per-thread CoInitialize bookkeeping)
# ─────────────────────────────────────────────────────────────────────────
class EnsureComInitTests(_BridgeBase):
    def test_initializes_once_per_thread(self):
        fake_pc = mock.MagicMock()
        ib._ensure_com_init(fake_pc)
        ib._ensure_com_init(fake_pc)     # second call is a no-op
        fake_pc.CoInitialize.assert_called_once()
        self.assertTrue(ib._itunes_com_tls.initialized)


# ─────────────────────────────────────────────────────────────────────────
# get_client — the gating logic (the heart of the module)
# ─────────────────────────────────────────────────────────────────────────
class GetClientGatingTests(_BridgeBase):
    def test_apple_music_active_short_circuits(self):
        ib.set_apple_music_active_check(lambda: True)
        # No COM modules injected — must not be needed (early return).
        app, err = ib.get_client(force=False)
        self.assertIsNone(app)
        self.assertIn("Apple Music is the active player", err)

    def test_apple_music_check_exception_treated_as_inactive(self):
        # Predicate raises → treated as "not active"; falls through to
        # is_running (False) + auto-launch off → friendly error.
        def boom():
            raise RuntimeError("predicate broke")
        ib.set_apple_music_active_check(boom)
        ib.set_auto_launch(False)
        self._inject("psutil", _fake_psutil([]))   # iTunes not running
        app, err = ib.get_client(force=False)
        self.assertIsNone(app)
        self.assertIn("auto-launch is disabled", err)

    def test_force_bypasses_apple_music_check(self):
        # force=True skips the Apple-Music guard; iTunes running → binds.
        ib.set_apple_music_active_check(lambda: True)
        self._inject("psutil", _fake_psutil(["iTunes.exe"]))
        app_obj = _ready_app()
        w, c, pc = _fake_com(app=app_obj)
        self._inject_com(w, c, pc)
        app, err = ib.get_client(force=True)
        self.assertIs(app, app_obj)
        self.assertIsNone(err)

    def test_not_running_no_autolaunch_returns_error_before_com(self):
        ib.set_auto_launch(False)
        self._inject("psutil", _fake_psutil([]))
        # Deliberately inject NO com modules — the guard must return before
        # importing win32com. If it tried, we'd get an ImportError instead.
        self._inject("win32com", None)
        self._inject("pythoncom", None)
        app, err = ib.get_client(force=False)
        self.assertIsNone(app)
        self.assertIn("auto-launch is disabled", err)

    def test_pywin32_absent_reported(self):
        # iTunes running, but win32com import fails → friendly pywin32 error.
        self._inject("psutil", _fake_psutil(["iTunes.exe"]))
        self._inject("win32com", None)
        self._inject("win32com.client", None)
        self._inject("pythoncom", None)
        with mock.patch.dict(sys.modules,
                             {"win32com": None, "win32com.client": None,
                              "pythoncom": None}):
            app, err = ib.get_client(force=False)
        self.assertIsNone(app)
        self.assertIn("pywin32 not installed", err)

    def test_running_get_active_object_success(self):
        self._inject("psutil", _fake_psutil(["iTunes.exe"]))
        app_obj = _ready_app()
        w, c, pc = _fake_com(app=app_obj)
        self._inject_com(w, c, pc)
        app, err = ib.get_client()
        self.assertIs(app, app_obj)
        self.assertIsNone(err)
        pc.CoInitialize.assert_called_once()

    def test_running_get_active_fails_dispatch_fallback_succeeds(self):
        self._inject("psutil", _fake_psutil(["iTunes.exe"]))
        app_obj = _ready_app()
        w, c, pc = _fake_com(app=app_obj,
                             get_active_exc=RuntimeError("ROT race"))
        self._inject_com(w, c, pc)
        app, err = ib.get_client()
        self.assertIs(app, app_obj)
        self.assertIsNone(err)

    def test_running_both_get_active_and_dispatch_fail(self):
        self._inject("psutil", _fake_psutil(["iTunes.exe"]))
        w, c, pc = _fake_com(app=None,
                             get_active_exc=RuntimeError("no ROT"),
                             dispatch_exc=RuntimeError("dispatch dead"))
        self._inject_com(w, c, pc)
        app, err = ib.get_client()
        self.assertIsNone(app)
        self.assertIn("could not connect to iTunes", err)

    def test_wait_for_ready_false_returns_immediately(self):
        self._inject("psutil", _fake_psutil(["iTunes.exe"]))
        app_obj = mock.MagicMock()   # library access would raise, but we skip it
        w, c, pc = _fake_com(app=app_obj)
        self._inject_com(w, c, pc)
        app, err = ib.get_client(wait_for_ready=False)
        self.assertIs(app, app_obj)
        self.assertIsNone(err)

    def test_wait_for_ready_times_out(self):
        self._inject("psutil", _fake_psutil(["iTunes.exe"]))
        app_obj = mock.MagicMock()
        # LibraryPlaylist access always raises → never ready → timeout path.
        type(app_obj).LibraryPlaylist = mock.PropertyMock(
            side_effect=RuntimeError("library loading"))
        w, c, pc = _fake_com(app=app_obj)
        self._inject_com(w, c, pc)
        # Drive the clock: first time() = start, subsequent jump past deadline.
        times = iter([1000.0, 1000.0, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(ib.time, "time", lambda: next(times)), \
                mock.patch.object(ib.time, "sleep"):
            app, err = ib.get_client(timeout=5.0)
        self.assertIsNone(app)
        self.assertIn("did not become ready", err)


class GetClientAutoLaunchTests(_BridgeBase):
    def test_autolaunch_spawns_then_dispatch_succeeds(self):
        # iTunes NOT running, auto-launch ON, force ON → launch() then Dispatch.
        ib.set_auto_launch(True)
        self._inject("psutil", _fake_psutil([]))     # not running
        app_obj = _ready_app()
        # GetActiveObject won't be called (running False). Dispatch succeeds.
        w, c, pc = _fake_com(app=app_obj)
        self._inject_com(w, c, pc)
        with mock.patch.object(ib, "launch", return_value=(True, None)), \
                mock.patch.object(ib.time, "sleep"), \
                mock.patch.object(ib.time, "time",
                                  side_effect=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]):
            app, err = ib.get_client(force=True)
        self.assertIs(app, app_obj)
        self.assertIsNone(err)

    def test_autolaunch_launch_fails(self):
        ib.set_auto_launch(True)
        self._inject("psutil", _fake_psutil([]))
        w, c, pc = _fake_com(app=None)
        self._inject_com(w, c, pc)
        with mock.patch.object(ib, "launch",
                               return_value=(False, "no exe")):
            app, err = ib.get_client(force=True)
        self.assertIsNone(app)
        self.assertIn("could not launch iTunes", err)

    def test_autolaunch_com_never_available(self):
        # launch() succeeds but Dispatch keeps failing until the 30s deadline.
        ib.set_auto_launch(True)
        self._inject("psutil", _fake_psutil([]))
        w, c, pc = _fake_com(app=None, dispatch_exc=RuntimeError("not up yet"))
        self._inject_com(w, c, pc)
        # time(): start for the launch loop, then jump past the 30s window.
        times = iter([0.0, 0.0, 100.0, 100.0])
        with mock.patch.object(ib, "launch", return_value=(True, None)), \
                mock.patch.object(ib.time, "sleep"), \
                mock.patch.object(ib.time, "time", lambda: next(times)):
            app, err = ib.get_client(force=True)
        self.assertIsNone(app)
        self.assertIn("COM never became available", err)

    def test_not_running_autolaunch_off_force_on_uses_outer_guard(self):
        # force=True bypasses the Apple-Music check but NOT the auto-launch
        # gate: with iTunes not running and auto-launch off, the OUTER guard
        # (the pre-COM-import bail) returns the friendly error before any COM
        # import. (The inner defence-in-depth twin at line 248-252 is only
        # reachable via a TOCTOU flip of the _AUTO_LAUNCH global between the two
        # checks — see the pragma there.)
        ib.set_auto_launch(False)
        ib.set_apple_music_active_check(None)
        self._inject("psutil", _fake_psutil([]))     # not running
        # No COM modules needed — the outer guard returns before importing them.
        app, err = ib.get_client(force=True)
        self.assertIsNone(app)
        self.assertIn("auto-launch is disabled", err)


if __name__ == "__main__":
    unittest.main()
