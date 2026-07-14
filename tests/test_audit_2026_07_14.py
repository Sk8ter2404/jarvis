"""Regression tests for the 2026-07-14 multi-agent audit (batch 1: the
crash-class and destroy-user-data findings).

Each test pins a defect that a 12-finder / 3-skeptic adversarial audit
confirmed against the live tree. They are grouped here (rather than scattered)
so the batch stays legible; each class names the finding it locks down.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import core.actions as A                       # noqa: E402
from audio import kinect_bridge as kb          # noqa: E402
from tests._skill_harness import load_skill_isolated  # noqa: E402


class KinectCloseIsFinalTests(unittest.TestCase):
    """Finding #11: close() was RESURRECTABLE. The body pump ticks at 30 Hz, so
    a tick already past its stop-check calls get_runtime(), finds _runtime[0]
    None but _ENABLED still True, and re-opens the sensor (spawning a fresh
    pump) — leaving a thread holding a live Kinect driver handle microseconds
    before TerminateProcess, i.e. the corpse class v2.0.57 exists to prevent."""

    def setUp(self):
        self._enabled = kb._ENABLED
        self._rt = kb._runtime[0]
        self.addCleanup(self._restore)

    def _restore(self):
        kb._ENABLED = self._enabled
        kb._runtime[0] = self._rt

    def test_close_final_clears_enabled(self):
        kb._ENABLED = True
        kb._runtime[0] = None
        with mock.patch.object(kb, "stop_body_pump"):
            kb.close(final=True)
        self.assertFalse(kb._ENABLED,
                         "a final close must make re-opening impossible")

    def test_plain_close_leaves_enabled(self):
        # The non-final close stays a soft release (a later action may re-open).
        kb._ENABLED = True
        kb._runtime[0] = None
        with mock.patch.object(kb, "stop_body_pump"):
            kb.close()
        self.assertTrue(kb._ENABLED)


class ReleaseNativeResourcesTests(unittest.TestCase):
    """Finding #11 (call site): the exit path must use the FINAL close."""

    def test_release_uses_final_close(self):
        # `from audio import kinect_bridge` resolves via the PACKAGE ATTRIBUTE,
        # not sys.modules, so patch the real functions in place.
        bc = mock.Mock()
        with mock.patch.object(kb, "close") as close, \
             mock.patch("core.voice_clone.unload") as unload:
            A._release_native_resources(bc)
        close.assert_called_once_with(final=True)
        unload.assert_called_once()
        bc.sd.stop.assert_called_once()
        bc._face_track_stop.set.assert_called_once()

    def test_release_falls_back_when_bridge_lacks_the_kwarg(self):
        # An older bridge without close(final=) must still be disabled, not
        # left enabled with a live pump.
        bc = mock.Mock()
        with mock.patch.object(kb, "close",
                               side_effect=TypeError("no kwarg")) as close, \
             mock.patch.object(kb, "set_enabled") as set_enabled, \
             mock.patch("core.voice_clone.unload"):
            A._release_native_resources(bc)
        close.assert_called_once_with(final=True)
        set_enabled.assert_called_once_with(False)


class UpgradeExitPathTests(unittest.TestCase):
    """Finding #7: _act_upgrade was the LAST os._exit in the tree — no native
    release, no singleton release, no clean flag, and the loader-lock deadlock
    that turns the process into an unkillable VRAM-pinning corpse."""

    def test_upgrade_source_has_no_raw_os_exit(self):
        import inspect
        src = inspect.getsource(A._act_upgrade)
        # Strip comments — the fix's own explanation names the old call.
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        self.assertNotIn("os._exit", code,
                         "the upgrade path must exit via _hard_exit_via_bc "
                         "(TerminateProcess), never a raw os._exit")
        self.assertIn("_hard_exit_via_bc", code)
        self.assertIn("_release_native_resources", code)
        # clean=True: upgrade_jarvis.py relaunches JARVIS, so the watchdog
        # must NOT also resurrect it (that would double-boot).
        self.assertIn("clean=True", code)


class YoutubeSearchClosesTabNotWindowTests(unittest.TestCase):
    """Finding #10: _close_prior_youtube_windows called w.close() on a
    TOP-LEVEL browser window. A window's title is its ACTIVE TAB's title, so an
    owner sitting on a YouTube tab in a 12-tab Chrome window lost ALL TWELVE
    tabs on every "play X on youtube"."""

    def _load(self):
        mod, _actions = load_skill_isolated("youtube_search")
        return mod

    def test_never_calls_window_close(self):
        mod = self._load()
        win = mock.Mock()
        win.title = "Take On Me - YouTube - Google Chrome"
        fake_gw = mock.Mock()
        fake_gw.getAllWindows.return_value = [win]
        fake_pag = mock.Mock()
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw,
                                           "pyautogui": fake_pag}), \
             mock.patch.object(mod.time, "sleep"):
            mod._close_prior_youtube_windows()
        win.close.assert_not_called()          # THE regression
        win.activate.assert_called_once()
        fake_pag.hotkey.assert_called_once_with("ctrl", "w")

    def test_activation_failure_touches_nothing(self):
        # If we cannot focus the window we must close NOTHING — an extra tab is
        # infinitely cheaper than the owner's lost tabs.
        mod = self._load()
        win = mock.Mock()
        win.title = "Take On Me - YouTube - Google Chrome"
        win.activate.side_effect = RuntimeError("no focus")
        fake_gw = mock.Mock()
        fake_gw.getAllWindows.return_value = [win]
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw}):
            mod._close_prior_youtube_windows()   # must not raise
        win.close.assert_not_called()

    def test_non_youtube_window_untouched(self):
        mod = self._load()
        win = mock.Mock()
        win.title = "Billing - Google Chrome"
        fake_gw = mock.Mock()
        fake_gw.getAllWindows.return_value = [win]
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw}):
            mod._close_prior_youtube_windows()
        win.close.assert_not_called()
        win.activate.assert_not_called()


class AmbientListenClaimsBeforeOpenTests(unittest.TestCase):
    """Finding #1: both ambient workers set the PortAudio ownership guard AFTER
    stream.start(), leaving a window where the callback was live but the guard
    read 0 — a concurrent _refresh_devices in that gap frees PortAudio under
    the live callback and heap-corrupts the process (0xc0000374). record_speech
    was fixed for exactly this; the ambient copies were not."""

    def test_source_claims_before_opening_the_stream(self):
        src = open(os.path.join(_PROJECT, "skills", "ambient_listen.py"),
                   encoding="utf-8").read()
        for anchor in ("sd.InputStream(",):
            self.assertIn(anchor, src)
        # In BOTH workers the claim must precede the InputStream construction.
        for start in range(len(src)):
            break
        claim = "_set_ambient_stream_active(True)"
        open_call = "stream = sd.InputStream("
        claims = [i for i in range(len(src)) if src.startswith(claim, i)]
        opens = [i for i in range(len(src)) if src.startswith(open_call, i)]
        self.assertEqual(len(opens), 2, "two workers open a dedicated stream")
        for o in opens:
            self.assertTrue(any(c < o for c in claims),
                            "ownership must be claimed BEFORE sd.InputStream()")
        # And every failure path must give it back (no stranded refcount).
        self.assertGreaterEqual(src.count("_set_ambient_stream_active(False)"), 3)


if __name__ == "__main__":
    unittest.main()
