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
from tests._monolith_harness import (                  # noqa: E402
    load_monolith, requires_monolith)


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

    def test_release_waits_for_the_in_flight_gpu_worker(self):
        # 2026-07-14: unload() cannot release a model that is still LOADING.
        # A restart 83s into boot (chatterbox mid-warm) corpsed the process
        # even WITH the v2.0.57 release, because TerminateProcess landed while
        # a thread was parked inside the CUDA driver. The teardown must wait
        # for the voice-clone single-flight guard to clear first.
        bc = mock.Mock()
        bc._voice_clone_inflight = [True]
        calls = []

        def _sleep(_):
            calls.append(1)
            if len(calls) >= 3:
                bc._voice_clone_inflight[0] = False   # worker finishes

        with mock.patch.object(kb, "close"), \
             mock.patch("core.voice_clone.unload") as unload, \
             mock.patch.object(A.time, "sleep", side_effect=_sleep), \
             mock.patch("builtins.print"):
            A._release_native_resources(bc)
        self.assertGreaterEqual(len(calls), 3, "it waited for the GPU worker")
        unload.assert_called_once()      # and only unloaded AFTER the wait

    def test_release_does_not_hang_on_a_stuck_worker(self):
        # A worker that never finishes must not hang the teardown — the wait is
        # bounded (the caller's failsafe timer is the final backstop).
        bc = mock.Mock()
        bc._voice_clone_inflight = [True]          # never clears
        t = {"now": 0.0}
        with mock.patch.object(kb, "close"), \
             mock.patch("core.voice_clone.unload") as unload, \
             mock.patch.object(A.time, "time", side_effect=lambda: t["now"]), \
             mock.patch.object(A.time, "sleep",
                               side_effect=lambda s: t.__setitem__("now",
                                                                   t["now"] + s)), \
             mock.patch("builtins.print"):
            A._release_native_resources(bc)
        unload.assert_called_once()      # gave up waiting and pressed on

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


@requires_monolith
class CameraProbeBudgetTests(unittest.TestCase):
    """Finding #2 (the two-day camera mystery). Callers probe "in parallel",
    but EVERY worker serializes on _camera_io_lock — and the joiner's clock
    started at THREAD start, not at lock acquisition. So the 2nd/3rd worker
    could burn its whole budget merely QUEUED and be reported "wedged" without
    ever calling cv2.VideoCapture: healthy cameras marked bad ("failed to open
    in 2.0s"). The budget must start when the work does."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_queued_worker_is_not_charged_for_lock_wait(self):
        import threading
        import time as _t
        bc = self.bc
        # Hold the camera I/O lock for longer than one probe budget, then free
        # it. The probe must WAIT (not fail) and then succeed on its own clock.
        released = threading.Event()

        def _hog():
            with bc._camera_io_lock:
                released.wait(timeout=5)

        hog = threading.Thread(target=_hog, daemon=True)
        hog.start()
        _t.sleep(0.2)                     # ensure the hog owns the lock

        fake_cap = mock.Mock()
        fake_cap.isOpened.return_value = True
        fake_cap.read.return_value = (True, "frame")
        with mock.patch.object(bc.cv2, "VideoCapture", return_value=fake_cap):
            t0 = _t.monotonic()
            # Free the lock shortly AFTER the probe's own (short) budget would
            # have expired if the queue time had been charged against it.
            threading.Timer(0.6, released.set).start()
            ok = bc._probe_camera_index(7, timeout_sec=0.4)
            dt = _t.monotonic() - t0
        self.assertTrue(ok, "a healthy camera queued behind the lock must NOT "
                            "be reported dead")
        self.assertGreater(dt, 0.5, "it really did wait for the lock")

    def test_unavailable_lock_reports_honestly(self):
        import threading
        import time as _t
        bc = self.bc
        forever = threading.Event()

        def _hog():
            with bc._camera_io_lock:
                forever.wait(timeout=30)

        hog = threading.Thread(target=_hog, daemon=True)
        hog.start()
        _t.sleep(0.2)
        try:
            with mock.patch.object(bc, "CAMERA_PROBE_MAX", 1), \
                 mock.patch("builtins.print"):
                # lock_budget = max(2.0, ...) → ~2s, then an honest False.
                ok = bc._probe_camera_index(7, timeout_sec=0.1)
            self.assertFalse(ok)
        finally:
            forever.set()


@requires_monolith
class SideTileNamesFollowConfigTests(unittest.TestCase):
    """Finding #3: the HUD composite pinned the LEFT tile to 'logi c270'. When
    the owner swapped that camera out, the tile drew a dim 'off' placeholder
    forever while the camera was open and streaming — a healthy camera reported
    as dead, silently. The needles must come from CAMERAS, the single source of
    truth."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_names_are_derived_from_cameras(self):
        bc = self.bc
        cams = [
            {"index": 2, "label": "Left webcam (left monitor)",
             "name": "emeet c960", "look_x": 0.5},
            {"index": 0, "label": "Right webcam (top of right monitor)",
             "name": "usb 2.0 camera", "look_x": 0.85},
        ]
        with mock.patch.object(bc, "CAMERAS", cams):
            names = bc._kinect_preview_webcam_names()
        self.assertEqual(names, {"left": "emeet c960",
                                 "right": "usb 2.0 camera"})

    def test_every_needle_matches_a_real_configured_camera(self):
        # The live rig's roster must always be self-consistent — this is the
        # assertion that would have caught the C270→C960 swap.
        bc = self.bc
        names = bc._kinect_preview_webcam_names()
        configured = {str(c.get("name") or "").lower() for c in bc.CAMERAS}
        for slot, needle in names.items():
            self.assertIn(needle, configured,
                          f"the {slot} tile looks for a camera that is not in "
                          f"CAMERAS — the C270→C960 rot class")


class PiiHookFailsClosedTests(unittest.TestCase):
    """Finding #19 (public repo!): check_no_pii deliberately exits 2 when it
    CANNOT SCAN SAFELY (a present-but-broken pii_local.py — the fail-closed
    path added precisely so a degraded scanner can't wave secrets through).
    The pre-commit hook only blocked on exit 1, so exit 2 fell into `exit 0`:
    the guard protecting a PUBLIC repo silently PASSED the commit in exactly
    the case it exists to catch."""

    def _hook_text(self):
        import tools.install_git_hooks as igh
        return igh.PRE_COMMIT_HOOK

    def test_blocks_on_any_nonzero_status(self):
        hook = self._hook_text()
        self.assertIn('-ne 0', hook,
                      "the hook must block on ANY nonzero status, not just 1")

    def test_still_degrades_when_python_is_absent(self):
        # 127 = command not found. Bricking every commit on a machine without
        # python would be worse than the warning; CI is the backstop there.
        self.assertIn('-eq 127', self._hook_text())

    def test_installed_hook_matches_the_template(self):
        import tools.install_git_hooks as igh
        path = os.path.join(igh.hooks_dir(), "pre-commit")
        if not os.path.exists(path):
            self.skipTest("no pre-commit hook installed in this checkout")
        live = open(path, encoding="utf-8").read()
        self.assertIn('-ne 0', live,
                      "the INSTALLED hook is stale — re-run "
                      "tools/install_git_hooks.py")


@requires_monolith
class BlindToggleTests(unittest.TestCase):
    """Finding #6: "playpause"/"space" are media-key TOGGLES — firing one at a
    video that is ALREADY PLAYING pauses it. They are only safe because a
    verify step reports the state first. When confirmation is STRUCTURALLY
    impossible (no title_confirm, no usable vision backend), every verify reads
    NOT-PLAYING and the retry loop presses the toggle at a player YouTube
    already autoplayed — the owner watched JARVIS pause the video it had just
    started, then report failure."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_confirmation_possible_with_title_confirm(self):
        self.assertTrue(
            self.bc._streaming_confirmation_possible({"title_confirm": True}))

    def test_confirmation_possible_with_vision(self):
        bc = self.bc
        with mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "_vision_click_backend_available",
                               return_value=True):
            self.assertTrue(bc._streaming_confirmation_possible({}))

    def test_confirmation_impossible_without_either(self):
        bc = self.bc
        with mock.patch.object(bc, "SCREEN_VISION_ENABLED", False):
            self.assertFalse(bc._streaming_confirmation_possible({}))

    def test_blind_toggles_are_dropped_when_unconfirmable(self):
        bc = self.bc
        cfg = {"play_strategies": ["recheck", "playpause", "space"],
               "verify_attempts": 1, "verify_wait": 0.0,
               "play_hint": None, "service_key": "youtube"}
        pressed = []
        with mock.patch.object(bc, "_streaming_confirmation_possible",
                               return_value=False), \
             mock.patch.object(bc, "_streaming_confirm_playback",
                               return_value=(False, "vision unavailable")), \
             mock.patch.object(bc, "_streaming_apply_play_strategy",
                               side_effect=lambda s, c, r: (pressed.append(s),
                                                            (True, s))[1]), \
             mock.patch.object(bc, "_streaming_go_fullscreen"), \
             mock.patch.object(bc.time, "sleep"), \
             mock.patch("builtins.print"):
            bc._streaming_play_and_verify(cfg, "YouTube", "q")
        self.assertNotIn("playpause", pressed,
                         "a blind toggle would PAUSE the playing video")
        self.assertNotIn("space", pressed)

    def test_toggles_are_kept_when_confirmation_works(self):
        bc = self.bc
        cfg = {"play_strategies": ["playpause"], "verify_attempts": 1,
               "verify_wait": 0.0, "play_hint": None, "service_key": "youtube"}
        pressed = []
        with mock.patch.object(bc, "_streaming_confirmation_possible",
                               return_value=True), \
             mock.patch.object(bc, "_streaming_confirm_playback",
                               return_value=(False, "not yet")), \
             mock.patch.object(bc, "_streaming_apply_play_strategy",
                               side_effect=lambda s, c, r: (pressed.append(s),
                                                            (True, s))[1]), \
             mock.patch.object(bc, "_streaming_go_fullscreen"), \
             mock.patch.object(bc.time, "sleep"), \
             mock.patch("builtins.print"):
            bc._streaming_play_and_verify(cfg, "YouTube", "q")
        self.assertIn("playpause", pressed,
                      "with a working verify, the toggle is safe and useful")


@requires_monolith
class LlmIndependentControlPlaneTests(unittest.TestCase):
    """2026-07-14, found by hitting it live: restart/shutdown were reachable
    ONLY as ACTIONS, which pass through intent classification — i.e. through
    the LOCAL BRAIN. So when the brain is starved (VRAM pressure) or wedged,
    the one command that would FIX it ("restart yourself") is precisely the
    command you cannot issue: JARVIS replies "my local model isn't responding"
    and stays broken. A control-plane operation must never depend on the thing
    it repairs. The tray channel is drained by a 2 Hz thread, so it works even
    while the main loop is blocked."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _dispatch(self, cmd):
        bc = self.bc
        with mock.patch("core.actions._act_restart") as restart, \
             mock.patch("core.actions._act_shutdown_jarvis") as shutdown, \
             mock.patch("builtins.print"):
            bc._dispatch_tray_command(cmd, {})
        return restart, shutdown

    def test_restart_runs_without_the_llm(self):
        restart, shutdown = self._dispatch("restart")
        restart.assert_called_once()
        shutdown.assert_not_called()

    def test_shutdown_runs_without_the_llm(self):
        restart, shutdown = self._dispatch("shutdown")
        shutdown.assert_called_once()
        restart.assert_not_called()

    def test_unknown_command_is_ignored(self):
        restart, shutdown = self._dispatch("definitely_not_a_command")
        restart.assert_not_called()
        shutdown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
