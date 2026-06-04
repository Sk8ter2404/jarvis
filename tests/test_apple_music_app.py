"""Tests for audio.apple_music_app — the lazy bridge to the new UWP Apple Music app.

The whole point of this module is to control the Store Apple Music app the only
LEGITIMATE way — launch via AUMID, drive transport with OS media keys (elsewhere),
and read the window title for a best-effort now-playing — while NEVER raising and
NEVER importing an optional dep (psutil / subprocess shell-outs / pygetwindow) at
module-import time in a way that can fail.

Everything external is faked:
  * psutil is injected into sys.modules so is_running / is_active_media_app run on
    any OS (including the bare-Linux CI runner where psutil is absent).
  * subprocess.run (Get-StartApps / Get-AppxPackage) and subprocess.Popen
    (explorer launch) are patched so nothing shells out or spawns.
  * pygetwindow is injected so now_playing parses a fake window title.

stdlib unittest + mock only.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from audio import apple_music_app as am


# ─────────────────────────────────────────────────────────────────────────
# fakes
# ─────────────────────────────────────────────────────────────────────────
def _fake_psutil(names):
    """psutil stand-in whose process_iter yields procs with the given names."""
    mod = types.ModuleType("psutil")

    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    class _Proc:
        def __init__(self, name):
            self.info = {"name": name}

    mod.NoSuchProcess = NoSuchProcess
    mod.AccessDenied = AccessDenied
    mod.process_iter = lambda _attrs: [_Proc(n) for n in names]
    return mod


class _FakeWin:
    def __init__(self, title):
        self.title = title


def _fake_pgw(titles):
    mod = types.ModuleType("pygetwindow")
    mod.getAllWindows = lambda: [_FakeWin(t) for t in titles]
    return mod


def _completed(stdout="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, returncode=returncode, stderr="")


class _Base(unittest.TestCase):
    def setUp(self):
        # Reset the AUMID cache between tests so resolution order is deterministic.
        am._aumid_cache[0] = None
        self.addCleanup(lambda: am._aumid_cache.__setitem__(0, None))

    def _inject(self, name, module):
        old = sys.modules.get(name)
        had = name in sys.modules
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

        def restore():
            if had:
                sys.modules[name] = old
            else:
                sys.modules.pop(name, None)
        self.addCleanup(restore)


# ─────────────────────────────────────────────────────────────────────────
# aumid
# ─────────────────────────────────────────────────────────────────────────
class AumidTests(_Base):
    def test_returns_known_when_startapps_empty(self):
        with mock.patch.object(am.subprocess, "run", return_value=_completed("")):
            self.assertEqual(am.aumid(), am._KNOWN_AUMID)

    def test_resolves_via_get_startapps_and_caches(self):
        resolved = "AppleInc.AppleMusicWin_otherhash!App"
        with mock.patch.object(am.subprocess, "run",
                               return_value=_completed(resolved + "\n")) as mrun:
            first = am.aumid()
        self.assertEqual(first, resolved)
        # Cached → a second call does NOT shell out again.
        with mock.patch.object(am.subprocess, "run",
                               side_effect=AssertionError("should be cached")):
            self.assertEqual(am.aumid(), resolved)
        mrun.assert_called_once()

    def test_startapps_multiline_takes_first(self):
        out = "\n  AppleInc.AppleMusicWin_aaa!App  \nsomething.else!App\n"
        with mock.patch.object(am.subprocess, "run", return_value=_completed(out)):
            self.assertEqual(am.aumid(), "AppleInc.AppleMusicWin_aaa!App")

    def test_known_not_cached_so_later_dynamic_wins(self):
        # First call: shell-out fails → falls back to known constant, NOT cached.
        with mock.patch.object(am.subprocess, "run",
                               side_effect=OSError("no powershell")):
            self.assertEqual(am.aumid(), am._KNOWN_AUMID)
        self.assertIsNone(am._aumid_cache[0])  # known constant was not cached
        # Later, the shell responds → dynamic value now wins.
        with mock.patch.object(am.subprocess, "run",
                               return_value=_completed("AppleInc.AppleMusicWin_z!App")):
            self.assertEqual(am.aumid(), "AppleInc.AppleMusicWin_z!App")

    def test_resolve_via_startapps_swallows_exception(self):
        with mock.patch.object(am.subprocess, "run",
                               side_effect=RuntimeError("boom")):
            self.assertIsNone(am._resolve_aumid_via_startapps())


# ─────────────────────────────────────────────────────────────────────────
# is_running / is_installed
# ─────────────────────────────────────────────────────────────────────────
class IsRunningTests(_Base):
    def test_true_when_process_present(self):
        self._inject("psutil", _fake_psutil(["chrome.exe", "AppleMusic.exe"]))
        self.assertTrue(am.is_running())

    def test_false_when_absent(self):
        self._inject("psutil", _fake_psutil(["chrome.exe", "explorer.exe"]))
        self.assertFalse(am.is_running())

    def test_case_insensitive(self):
        self._inject("psutil", _fake_psutil(["APPLEMUSIC.EXE"]))
        self.assertTrue(am.is_running())

    def test_psutil_absent_returns_false(self):
        self._inject("psutil", None)
        with mock.patch.dict(sys.modules, {"psutil": None}):
            self.assertFalse(am.is_running())

    def test_none_name_tolerated(self):
        self._inject("psutil", _fake_psutil([None, "AppleMusic.exe"]))
        self.assertTrue(am.is_running())


class IsInstalledTests(_Base):
    def test_true_when_appx_present(self):
        with mock.patch.object(am.subprocess, "run",
                               return_value=_completed("AppleInc.AppleMusicWin\n")):
            self.assertTrue(am.is_installed())

    def test_false_when_empty(self):
        with mock.patch.object(am.subprocess, "run", return_value=_completed("")):
            self.assertFalse(am.is_installed())

    def test_false_and_no_raise_when_powershell_missing(self):
        with mock.patch.object(am.subprocess, "run",
                               side_effect=FileNotFoundError("no powershell")):
            self.assertFalse(am.is_installed())


# ─────────────────────────────────────────────────────────────────────────
# launch / ensure_running
# ─────────────────────────────────────────────────────────────────────────
class LaunchTests(_Base):
    def test_launch_calls_explorer_with_aumid(self):
        with mock.patch.object(am, "aumid", return_value="AID!App"), \
                mock.patch.object(am.subprocess, "Popen") as mpopen:
            ok, err = am.launch()
        self.assertTrue(ok)
        self.assertIsNone(err)
        mpopen.assert_called_once()
        argv = mpopen.call_args[0][0]
        self.assertEqual(argv[0], "explorer.exe")
        self.assertEqual(argv[1], r"shell:AppsFolder\AID!App")

    def test_launch_failure_reported(self):
        with mock.patch.object(am, "aumid", return_value="AID!App"), \
                mock.patch.object(am.subprocess, "Popen",
                                  side_effect=OSError("denied")):
            ok, err = am.launch()
        self.assertFalse(ok)
        self.assertIn("failed to launch Apple Music", err)
        self.assertIn("denied", err)

    def test_ensure_running_already_running_no_launch(self):
        with mock.patch.object(am, "is_running", return_value=True), \
                mock.patch.object(am, "launch",
                                  side_effect=AssertionError("must not launch")):
            ok, err = am.ensure_running()
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_ensure_running_launches_then_appears(self):
        # ensure_running imports `time` lazily; patch time.sleep at the stdlib
        # module so the poll loop doesn't actually wait.
        states = iter([False, False, True])  # not running, launch, then appears
        import time as _time
        with mock.patch.object(am, "is_running",
                               side_effect=lambda: next(states)), \
                mock.patch.object(am, "launch", return_value=(True, None)) as ml, \
                mock.patch.object(_time, "sleep", return_value=None):
            ok, err = am.ensure_running(timeout=5.0)
        self.assertTrue(ok)
        ml.assert_called_once()

    def test_ensure_running_launch_fails(self):
        with mock.patch.object(am, "is_running", return_value=False), \
                mock.patch.object(am, "launch", return_value=(False, "no exe")):
            ok, err = am.ensure_running()
        self.assertFalse(ok)
        self.assertEqual(err, "no exe")


# ─────────────────────────────────────────────────────────────────────────
# now_playing / is_active_media_app
# ─────────────────────────────────────────────────────────────────────────
class NowPlayingTests(_Base):
    def test_parses_track_from_window_title(self):
        self._inject("pygetwindow",
                     _fake_pgw(["Bohemian Rhapsody - Apple Music"]))
        self.assertEqual(am.now_playing(), "Bohemian Rhapsody")

    def test_strips_em_dash_decorator(self):
        self._inject("pygetwindow",
                     _fake_pgw(["Smooth Criminal — Apple Music"]))
        self.assertEqual(am.now_playing(), "Smooth Criminal")

    def test_bare_app_name_returns_none(self):
        self._inject("pygetwindow", _fake_pgw(["Apple Music"]))
        self.assertIsNone(am.now_playing())

    def test_no_music_window_returns_none(self):
        self._inject("pygetwindow", _fake_pgw(["Visual Studio Code", "Chrome"]))
        self.assertIsNone(am.now_playing())

    def test_pygetwindow_absent_returns_none(self):
        self._inject("pygetwindow", None)
        with mock.patch.dict(sys.modules, {"pygetwindow": None}):
            self.assertIsNone(am.now_playing())

    def test_title_with_no_decorator_returned_as_is(self):
        # A window that contains "apple music" but isn't just the bare name and
        # has no recognised decorator → surfaced as-is (best effort).
        self._inject("pygetwindow", _fake_pgw(["Now in Apple Music: Thriller"]))
        self.assertEqual(am.now_playing(), "Now in Apple Music: Thriller")

    def test_never_raises_on_window_error(self):
        bad = types.ModuleType("pygetwindow")

        def boom():
            raise RuntimeError("win32 boom")
        bad.getAllWindows = boom
        self._inject("pygetwindow", bad)
        self.assertIsNone(am.now_playing())   # swallowed → None


class IsActiveMediaAppTests(_Base):
    def test_true_when_running(self):
        with mock.patch.object(am, "is_running", return_value=True):
            self.assertTrue(am.is_active_media_app())

    def test_true_when_window_present_even_if_not_running(self):
        with mock.patch.object(am, "is_running", return_value=False):
            self._inject("pygetwindow", _fake_pgw(["X - Apple Music"]))
            self.assertTrue(am.is_active_media_app())

    def test_false_when_neither(self):
        with mock.patch.object(am, "is_running", return_value=False):
            self._inject("pygetwindow", _fake_pgw(["Chrome", "Notepad"]))
            self.assertFalse(am.is_active_media_app())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
