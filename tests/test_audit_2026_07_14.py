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


@requires_monolith
class VlmResidentExemptionTests(unittest.TestCase):
    """Finding #23. Since the v2.0.33 overhaul chat and vision are the SAME
    multimodal model, so a vision call usually needs ZERO new VRAM — the weights
    are already on the card. The co-load gates couldn't tell 'load a 9 GB model'
    from 'reuse the 9 GB model sitting right there', saw a nearly-full 24 GB
    card, and refused. Watched live in the sweep log: "REFUSING co-load: only
    3976 MB free ... for gemma4:12b" — with gemma4:12b resident."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _ps(self, *names):
        return [{"name": n, "size_vram": 9_000_000_000} for n in names]

    def test_resident_vlm_is_detected(self):
        bc = self.bc
        with mock.patch.object(bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(bc, "_ollama_loaded_models",
                               return_value=self._ps("gemma4:12b")):
            self.assertTrue(bc._local_vision_model_already_resident())

    def test_resident_match_is_on_the_base_tag(self):
        bc = self.bc
        with mock.patch.object(bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(bc, "_ollama_loaded_models",
                               return_value=self._ps("gemma4:12b-it-q4")):
            self.assertTrue(bc._local_vision_model_already_resident())

    def test_other_model_resident_is_not_the_vlm(self):
        bc = self.bc
        with mock.patch.object(bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(bc, "_ollama_loaded_models",
                               return_value=self._ps("qwen2.5:14b")):
            self.assertFalse(bc._local_vision_model_already_resident())

    def test_full_card_does_not_refuse_an_already_resident_vlm(self):
        """THE BUG: 3976 MB free < 7500 MB needed, yet the model is loaded. The
        call must go through — there is nothing to co-load."""
        bc = self.bc
        posted = {}

        def _fake_post(url, json=None, timeout=None):
            posted["timeout"] = timeout
            r = mock.Mock()
            r.ok = True
            r.json.return_value = {"message": {"content": "I see a terminal."}}
            return r

        with mock.patch.object(bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(bc, "_local_vision_usable", return_value=True), \
             mock.patch.object(bc, "_ollama_alive", return_value=True), \
             mock.patch.object(bc, "_ollama_has_model", return_value=True), \
             mock.patch.object(bc, "_ollama_loaded_models",
                               return_value=self._ps("gemma4:12b")), \
             mock.patch.object(bc, "_cuda0_free_vram_mb", return_value=3976), \
             mock.patch.object(bc.requests, "post", side_effect=_fake_post):
            out = bc._call_local_vision("what is on screen?", [b"png"])
        self.assertEqual(out, "I see a terminal.",
                         "an ALREADY-RESIDENT VLM must not be refused for lack "
                         "of free VRAM — it needs none")

    def test_full_card_still_refuses_a_cold_vlm(self):
        """The guard must NOT be gutted: a model that is genuinely not loaded
        still gets refused on a full card. That was the whole point of it."""
        bc = self.bc
        with mock.patch.object(bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(bc, "_local_vision_usable", return_value=True), \
             mock.patch.object(bc, "_ollama_alive", return_value=True), \
             mock.patch.object(bc, "_ollama_has_model", return_value=True), \
             mock.patch.object(bc, "_ollama_loaded_models", return_value=[]), \
             mock.patch.object(bc, "_cuda0_free_vram_mb", return_value=3976), \
             mock.patch.object(bc.requests, "post") as post:
            out = bc._call_local_vision("what is on screen?", [b"png"])
        self.assertIsNone(out)
        post.assert_not_called()


@requires_monolith
class LocalVisionTimeoutTests(unittest.TestCase):
    """Finding #14. ask_vision runs on the MAIN VOICE THREAD for every caller
    but _glance, and the local-VLM POST used a flat timeout=180 — so a wedged
    runner could leave JARVIS deaf and mute for three solid minutes. The 180 s
    only ever existed to cover a COLD weight load; a warm call needs a bound
    sized to decode."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _run_with(self, loaded):
        bc = self.bc
        seen = {}

        def _fake_post(url, json=None, timeout=None):
            seen["timeout"] = timeout
            r = mock.Mock()
            r.ok = True
            r.json.return_value = {"message": {"content": "ok"}}
            return r

        models = ([{"name": "gemma4:12b", "size_vram": 9_000_000_000}]
                  if loaded else [])
        with mock.patch.object(bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(bc, "_local_vision_usable", return_value=True), \
             mock.patch.object(bc, "_ollama_alive", return_value=True), \
             mock.patch.object(bc, "_ollama_has_model", return_value=True), \
             mock.patch.object(bc, "_ollama_loaded_models", return_value=models), \
             mock.patch.object(bc, "_cuda0_free_vram_mb", return_value=24000), \
             mock.patch.object(bc.requests, "post", side_effect=_fake_post):
            bc._call_local_vision("q", [b"png"])
        return seen["timeout"]

    def test_warm_call_gets_the_voice_safe_bound(self):
        self.assertEqual(self._run_with(loaded=True),
                         self.bc._LOCAL_VISION_TIMEOUT_WARM_S)

    def test_cold_load_still_gets_the_long_rope(self):
        self.assertEqual(self._run_with(loaded=False),
                         self.bc._LOCAL_VISION_TIMEOUT_COLD_S)

    def test_warm_bound_cannot_silence_jarvis_for_minutes(self):
        bc = self.bc
        self.assertLessEqual(bc._LOCAL_VISION_TIMEOUT_WARM_S, 60,
                             "the steady-state bound is how long JARVIS can go "
                             "deaf on a wedged runner — keep it short")
        self.assertLess(bc._LOCAL_VISION_TIMEOUT_WARM_S,
                        bc._LOCAL_VISION_TIMEOUT_COLD_S)


@requires_monolith
class AnthropicRetryBoundTests(unittest.TestCase):
    """Finding #22. `timeout` in the Anthropic SDK is PER ATTEMPT and the SDK
    default is max_retries=2 — so the documented "30 s" bound on the voice
    thread was really ~92 s. The 2026-05-30 pass even NAMED the multiplier in
    its comment ("× 2 retries") and then pinned only the timeout."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_client_pins_max_retries(self):
        bc = self.bc
        fake = mock.Mock()
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            bc._anthropic_client()
        _args, kwargs = fake.Anthropic.call_args
        self.assertEqual(kwargs["timeout"], bc._ANTHROPIC_TIMEOUT_S)
        self.assertEqual(kwargs["max_retries"], bc._ANTHROPIC_MAX_RETRIES)

    def test_retry_budget_is_small(self):
        self.assertLessEqual(self.bc._ANTHROPIC_MAX_RETRIES, 1)

    def test_no_bare_anthropic_constructions_remain(self):
        """The retry multiplier survived a pass that had already spotted it
        because the construction was duplicated eight times. Keep it funnelled:
        the ONLY anthropic.Anthropic( in the monolith is the factory's own."""
        import ast
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        # AST, not grep: a text scan also matches the factory's own docstring,
        # which talks ABOUT anthropic.Anthropic(...). Count real Call nodes.
        hits = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "Anthropic"]
        self.assertEqual(len(hits), 1,
                         f"expected only the _anthropic_client() factory to "
                         f"construct a client; found constructions at lines "
                         f"{hits}")

    def test_llm_client_module_pins_retries_too(self):
        from core import llm_client
        fake = mock.Mock()
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            llm_client._client(12.0)
        _args, kwargs = fake.Anthropic.call_args
        self.assertEqual(kwargs["max_retries"], llm_client.DEFAULT_MAX_RETRIES)
        self.assertLessEqual(llm_client.DEFAULT_MAX_RETRIES, 1)


class LiveBackendReportingTests(unittest.TestCase):
    """Finding #8. show_llm_stats / latency_benchmark imported AI_BACKEND and
    OLLAMA_MODEL from core.config AT CALL TIME — i.e. the BOOT values. switch_llm
    deliberately mutates only the monolith's globals (its own docstring says
    "nothing reads from core.config at runtime"; these two did). And
    core.config.OLLAMA_MODEL is still the shipped default "llama3" — a tag
    RETIRED from this box — so on the local backend they didn't merely name the
    wrong model, they named one that isn't installed."""

    def _bc(self, backend, live_local="gemma4:12b"):
        bc = mock.Mock()
        bc.AI_BACKEND = backend
        bc.CLAUDE_MODEL = "claude-sonnet-5"
        bc.OLLAMA_MODEL = "llama3"          # the frozen boot landmine
        bc._get_local_llm_model.return_value = live_local
        return bc

    def test_reports_the_live_backend_after_a_switch(self):
        with mock.patch.object(A, "_bc", return_value=self._bc("ollama")):
            backend, model = A._live_backend_and_model()
        self.assertEqual(backend, "ollama")
        self.assertEqual(model, "gemma4:12b")
        self.assertNotEqual(model, "llama3", "must not report the retired tag")

    def test_claude_backend_reports_the_claude_model(self):
        with mock.patch.object(A, "_bc", return_value=self._bc("claude")):
            backend, model = A._live_backend_and_model()
        self.assertEqual((backend, model), ("claude", "claude-sonnet-5"))

    def test_show_llm_stats_names_the_live_local_model(self):
        with mock.patch.object(A, "_bc", return_value=self._bc("ollama")):
            out = A._act_show_llm_stats("")
        self.assertIn("backend=ollama", out)
        self.assertIn("gemma4:12b", out)
        self.assertNotIn("llama3", out)

    def test_benchmark_resolves_the_backend_after_the_call_not_at_dispatch(self):
        """The old closure captured the backend at DISPATCH while _llm_quick
        picks it LIVE — so a local round-trip could be timed and then labelled
        "claude/...". Flip the backend mid-call; the label must follow."""
        bc = self._bc("claude")
        state = {"backend": "claude"}
        bc.AI_BACKEND = "claude"

        def _quick(**_kw):
            state["backend"] = "ollama"
            bc.AI_BACKEND = "ollama"        # a switch landed while we were out
            return "pong"

        bc._llm_quick.side_effect = _quick
        captured = {}
        bc._tray_async.side_effect = lambda _n, fn: captured.update(out=fn())
        with mock.patch.object(A, "_bc", return_value=bc):
            A._act_latency_benchmark("")
        self.assertIn("ollama/gemma4:12b", captured["out"],
                      "the benchmark must name the brain it ACTUALLY timed")


@requires_monolith
class GestureInterruptIsNotAcousticTests(unittest.TestCase):
    """Finding #20. The SWIPE gesture set `_barge_in_interrupted`, whose only
    speaker-stopping reader (_barge_watch) is started ONLY when a barge-in mic
    stream exists — i.e. BARGE_IN_ENABLED *and* a HEADSET. On speakers the
    gesture set a flag nobody would ever read. And it can't simply call
    request_tts_interrupt() either: both of that function's gates exist because
    the MIC HEARS THE SPEAKERS, and a hand in front of a depth sensor cannot be
    an acoustic echo."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_non_acoustic_interrupt_bypasses_the_echo_gate(self):
        bc = self.bc
        with mock.patch.object(bc, "_tts_playback_active", [True]), \
             mock.patch.object(bc, "_tts_current_text",
                               ["certainly sir, jarvis here"]), \
             mock.patch.object(bc, "_tts_interrupt") as ev, \
             mock.patch.object(bc, "_tts_interrupt_seq", [0]):
            ok = bc.request_tts_interrupt(source="kinect-swipe", acoustic=False)
        self.assertTrue(ok, "a hand swipe is not an echo of our own speakers")
        ev.set.assert_called_once()

    def test_non_acoustic_interrupt_ignores_the_wake_word_knob(self):
        bc = self.bc
        with mock.patch.object(bc, "_barge_in_wake_enabled", return_value=False), \
             mock.patch.object(bc, "_tts_playback_active", [True]), \
             mock.patch.object(bc, "_tts_current_text", ["hello"]), \
             mock.patch.object(bc, "_tts_interrupt") as ev, \
             mock.patch.object(bc, "_tts_interrupt_seq", [0]):
            ok = bc.request_tts_interrupt(source="kinect-swipe", acoustic=False)
        self.assertTrue(ok, "BARGE_IN_ENABLED is the WAKE-WORD knob; it must "
                            "not silently disable hand gestures")
        ev.set.assert_called_once()

    def test_acoustic_wake_path_is_unchanged(self):
        """The mic-facing gates must survive intact for the wake word."""
        bc = self.bc
        with mock.patch.object(bc, "_barge_in_wake_enabled", return_value=True), \
             mock.patch.object(bc, "_tts_playback_active", [True]), \
             mock.patch.object(bc, "_tts_current_text", ["yes, jarvis here"]), \
             mock.patch.object(bc, "_tts_interrupt") as ev:
            self.assertFalse(bc.request_tts_interrupt(source="wake-word"))
        ev.set.assert_not_called()

    def test_nothing_playing_is_still_refused(self):
        bc = self.bc
        with mock.patch.object(bc, "_tts_playback_active", [False]), \
             mock.patch.object(bc, "_tts_interrupt") as ev:
            self.assertFalse(
                bc.request_tts_interrupt(source="kinect-swipe", acoustic=False))
        ev.set.assert_not_called()

    def test_swipe_routes_through_the_live_mechanism(self):
        """The gesture must call request_tts_interrupt, NOT poke the dead flag."""
        kg, _actions = load_skill_isolated("kinect_gestures")
        bc = mock.Mock()
        bc.request_tts_interrupt.return_value = True
        bc._pending_confirmation = None
        kg._do_swipe(bc)
        bc.request_tts_interrupt.assert_called_once()
        _a, kwargs = bc.request_tts_interrupt.call_args
        self.assertFalse(kwargs["acoustic"],
                         "a depth-sensor swipe is not an acoustic barge-in")


@requires_monolith
class GlanceGateIsNotClaudeOnlyTests(unittest.TestCase):
    """Finding #38. `_vision_click_backend_available()` was written specifically
    to kill the `AI_BACKEND == "claude"` gate shape, and every vision call site
    migrated to it — except this one. So on a local-only box the glance
    fast-path was silently, permanently dead: the turn fell through to the LLM
    with NO screenshot and JARVIS answered about nothing. The irony is that
    ask_vision — the function this gate guards — handles the local backend
    itself and would have answered from the VLM."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_glance_runs_on_a_local_only_box(self):
        bc = self.bc
        with mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "_vision_click_backend_available",
                               return_value=True), \
             mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=True), \
             mock.patch.object(bc, "_focus_changed_recently", return_value=True), \
             mock.patch.object(bc, "_capture_focused_window_png",
                               return_value=None) as cap:
            bc.maybe_glance_response("what's this?")
        cap.assert_called_once()   # got PAST the gate; used to return None here

    def test_glance_still_refused_when_no_vision_backend(self):
        bc = self.bc
        with mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "_vision_click_backend_available",
                               return_value=False), \
             mock.patch.object(bc, "_is_glance_ambiguous_question",
                               return_value=True), \
             mock.patch.object(bc, "_focus_changed_recently", return_value=True), \
             mock.patch.object(bc, "_capture_focused_window_png") as cap:
            self.assertIsNone(bc.maybe_glance_response("what's this?"))
        cap.assert_not_called()

    def test_no_claude_only_vision_gate_survives(self):
        """The whole point of the predicate is that this shape stops recurring."""
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            offenders = [
                (i, ln.strip()) for i, ln in enumerate(fh, 1)
                if 'AI_BACKEND != "claude"' in ln
                and "SCREEN_VISION_ENABLED" in ln
                and not ln.lstrip().startswith("#")
            ]
        self.assertEqual(offenders, [],
                         f"a Claude-only vision gate came back: {offenders}")


class CreateSkillGatesOnLiveBackendTests(unittest.TestCase):
    """Finding #26 — the same stale read as #8, with teeth. switch_llm only
    writes bc.AI_BACKEND, so reading core.config here was wrong in BOTH
    directions: after "switch to local" the frozen "claude" let the handler
    through and it went on spending Claude credits authoring the skill — exactly
    what the switch was meant to prevent — while a box whose user_settings pin
    "ollama" could never create a skill even with Claude live."""

    def _bc(self, backend):
        bc = mock.Mock()
        bc.AI_BACKEND = backend
        bc.CLAUDE_MODEL = "claude-sonnet-5"
        bc.OLLAMA_MODEL = "llama3"
        bc._get_local_llm_model.return_value = "gemma4:12b"
        return bc

    def test_refuses_after_a_switch_to_local(self):
        """core.config still says "claude"; the live brain is ollama. Refuse —
        and, critically, do not spend a single cloud token."""
        with mock.patch.object(A, "_bc", return_value=self._bc("ollama")), \
             mock.patch("core.config.SKILLS_ENABLED", True), \
             mock.patch("core.config.AI_BACKEND", "claude"):
            out = A._act_create_skill("thing | do a thing")
        self.assertIn("Claude backend", out)

    def test_allows_when_claude_is_live_despite_a_frozen_ollama_config(self):
        bc = self._bc("claude")
        with mock.patch.object(A, "_bc", return_value=bc), \
             mock.patch("core.config.SKILLS_ENABLED", True), \
             mock.patch("core.config.AI_BACKEND", "ollama"):
            out = A._act_create_skill("no-pipe-so-it-stops-early")
        self.assertNotIn("requires SKILLS_ENABLED", out,
                         "Claude IS live; the frozen boot config must not veto")


class VolumeGrammarTests(unittest.TestCase):
    """Finding #27. set_volume is real, registered, and was written BECAUSE
    "set the volume to 30 percent" had no matching action. But it was only ever
    taught to the LOCAL cheatsheet — the default Claude backend's action grammar
    never listed it, so on the default brain that phrase still degraded to a
    single volume_down nudge: the exact bug the action was added to fix."""

    def test_claude_grammar_offers_absolute_volume(self):
        from core import prompts
        p = prompts.PC_CONTROL_PROMPT
        self.assertIn("set_volume", p,
                      "the default backend's action grammar must list set_volume")
        self.assertIn("[ACTION: set_volume, 30]", p,
                      "and show an absolute-volume example")


@requires_monolith
class ClaudeOptionalIsWiredTests(unittest.TestCase):
    """Finding #35/#39. CLAUDE_OPTIONAL shipped as a documented Settings toggle
    ("Claude is optional — never required"), persisted to user_settings.json,
    and was read by NOTHING: the preflight hard-coded the True behaviour. So
    unticking it — the owner explicitly asking JARVIS to insist on a working
    key — changed absolutely nothing."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_constant_has_a_real_consumer(self):
        import re
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            body = [ln for ln in fh
                    if re.search(r"\bCLAUDE_OPTIONAL\b", ln)
                    and not ln.lstrip().startswith("#")]
        self.assertTrue(body, "CLAUDE_OPTIONAL is dead config again")


class NightOwlManualOffSticksTests(unittest.TestCase):
    """Finding #30. Between 23:00 and 06:00 night-owl mode could NOT be turned
    off. _exit_night_owl cleared the active flag and recorded nothing, so the
    next watcher tick (<=60 s) saw `in_window and not active`, re-engaged,
    re-dimmed TTS + overlay, re-installed the nudge suppressors and re-announced
    "It's past 11, sir." The user's decision survived less than a minute, every
    single time. A reconciliation loop needs somewhere to record "the user
    overrode me"; this one had nowhere."""

    def setUp(self):
        self.mod, _actions = load_skill_isolated("night_owl_mode")
        self.mod._opted_out_night[0] = ""
        self.mod._night_owl_active[0] = False

    def test_night_key_is_stable_across_midnight(self):
        from datetime import datetime as dt
        m = self.mod
        evening = m._night_key(dt(2026, 7, 14, 23, 30))
        small_hours = m._night_key(dt(2026, 7, 15, 0, 30))
        self.assertEqual(evening, small_hours,
                         "23:30 and 00:30 are the SAME night — the calendar "
                         "date is not a usable key for a wrapping window")

    def test_manual_off_inside_the_window_suppresses_auto_reengage(self):
        from datetime import datetime as dt
        m = self.mod
        inside = dt(2026, 7, 15, 0, 30)     # small hours, inside the window
        with mock.patch.object(m, "_in_night_window", return_value=True), \
             mock.patch.object(m, "_night_key", return_value="2026-07-14"), \
             mock.patch.object(m, "_restore_tts_modifier"), \
             mock.patch.object(m, "_restore_nudge_suppressors"), \
             mock.patch.object(m, "_restore_prompt_addendum"), \
             mock.patch.object(m, "_set_overlay_dim"), \
             mock.patch.object(m, "_enqueue_speech"):
            m._night_owl_active[0] = True
            m._exit_night_owl(trigger="manual")
            self.assertFalse(m.is_night_owl_active())
            self.assertTrue(m._opted_out_of_this_night(inside),
                            "a manual off inside the window must be REMEMBERED "
                            "or the watcher undoes it within 60 seconds")

    def test_auto_morning_release_is_not_an_optout(self):
        """The 06:00 release is the window ENDING, not the user opting out —
        it must not suppress tomorrow night."""
        m = self.mod
        with mock.patch.object(m, "_in_night_window", return_value=True), \
             mock.patch.object(m, "_restore_tts_modifier"), \
             mock.patch.object(m, "_restore_nudge_suppressors"), \
             mock.patch.object(m, "_restore_prompt_addendum"), \
             mock.patch.object(m, "_set_overlay_dim"), \
             mock.patch.object(m, "_enqueue_speech"):
            m._night_owl_active[0] = True
            m._exit_night_owl(trigger="auto_morning")
        self.assertEqual(m._opted_out_night[0], "")

    def test_optout_expires_with_the_night(self):
        from datetime import datetime as dt
        m = self.mod
        m._opted_out_night[0] = "2026-07-14"
        self.assertTrue(m._opted_out_of_this_night(dt(2026, 7, 15, 2, 0)))
        # Next evening is a NEW night — auto-engage must work again.
        self.assertFalse(m._opted_out_of_this_night(dt(2026, 7, 15, 23, 30)))

    def test_turning_it_back_on_retracts_the_optout(self):
        m = self.mod
        m._opted_out_night[0] = "2026-07-14"
        with mock.patch.object(m, "_install_tts_modifier"), \
             mock.patch.object(m, "_install_nudge_suppressors"), \
             mock.patch.object(m, "_apply_prompt_addendum"), \
             mock.patch.object(m, "_set_overlay_dim"), \
             mock.patch.object(m, "_enqueue_speech"), \
             mock.patch.object(m, "_in_night_window", return_value=True):
            m._enter_night_owl(trigger="manual")
        self.assertEqual(m._opted_out_night[0], "")


class StreamingFailuresAreSpokenTests(unittest.TestCase):
    """Finding #24. The streaming capability-gate returns are failure-SHAPED but
    carried no canonical FAILURE_MARKER, and the streaming actions sit in
    neither INFORMATIVE_ACTIONS nor SPEAK_RESULT_VERBATIM_ACTIONS. So with
    SCREEN_VISION_ENABLED off (a supported config — the gate exists for it), the
    user heard only the inline "Of course, sir", stared at a Netflix search page,
    and was NEVER told why nothing played. Sibling strings that DO carry a marker
    ("couldn't see the first result") prove the intended mechanism."""

    def _is_failure(self, result: str) -> bool:
        from core.failure_markers import FAILURE_MARKERS
        low = (result or "").lower()
        return any(m in low for m in FAILURE_MARKERS)

    def test_every_streaming_gate_return_is_failure_classified(self):
        """The ASSEMBLED strings, as the user's ears would receive them. (Not a
        source grep — these are built from split f-string literals, so no single
        fragment appears contiguously in the file.)"""
        assembled = [
            "opened Netflix for 'Stranger Things', but I couldn't start "
            "playback — auto-play needs UI automation (keyboard control)",
            "opened Netflix search for 'x', but I couldn't select the first "
            "result — auto-click needs SCREEN_VISION_ENABLED + "
            "UI_AUTOMATION_ENABLED + a vision backend",
            "opened Spotify search for 'x', but I couldn't select the first "
            "result — UI automation is unavailable; you may need to click it "
            "yourself",
            "opened Apple Music Library > Playlists, but I couldn't start the "
            "playlist — auto-click needs SCREEN_VISION_ENABLED",
            "opened YouTube search for 'x', but I couldn't continue — failsafe",
            "play attempt on Netflix failed: boom",
        ]
        for s in assembled:
            self.assertTrue(self._is_failure(s),
                            f"matches no FAILURE_MARKER, so it reaches neither "
                            f"the follow-up loop nor the verbatim speak-set — "
                            f"the user would hear nothing: {s[:60]!r}")

    def test_the_old_silent_phrasings_are_gone(self):
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        # The OLD, marker-free constructions, as they appeared verbatim in the
        # source. (Deliberately anchored on the leading `'{q}' —` / `but {e}`
        # shape: the REPLACEMENTS legitimately still mention "auto-play needs
        # UI automation" as the explanation — what changed is that they now lead
        # with "but I couldn't …", which is what carries the marker.)
        for dead in ("'{q}' — auto-play needs UI ",
                     "aborted: {e}",
                     "search for '{q}' but {e}"):
            self.assertNotIn(dead, src,
                             f"unmarked failure phrasing came back: {dead!r}")

    def test_a_marker_free_phrasing_would_actually_fail_this_guard(self):
        """Keep the guard honest — the OLD wording must classify as non-failure,
        proving these assertions aren't vacuously true."""
        old = ("opened Netflix for 'x' — auto-play needs UI automation "
               "(keyboard control) to start playback")
        self.assertFalse(self._is_failure(old),
                         "the pre-fix wording should be the silent case")


@requires_monolith
class AskVisionImportFallbackTests(unittest.TestCase):
    """Finding #42. `import anthropic` lived INSIDE the try whose except headers
    say `except anthropic.APIStatusError`. If the import fails, Python still has
    to EVALUATE that header to see if it matches — and `anthropic` is unbound, so
    that raises NameError, which propagates straight out (an exception raised
    while matching an EARLIER except never reaches a LATER `except Exception`).
    The catch-all's own comment CLAIMED it handled a failed import; it never
    could. A missing SDK must degrade to the local VLM, not crash the action."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_missing_sdk_falls_back_to_local_not_crash(self):
        bc = self.bc
        with mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch("core.config.model_route", return_value="auto"), \
             mock.patch.object(bc, "_call_local_vision", return_value="a terminal"), \
             mock.patch.dict(sys.modules, {"anthropic": None}):
            # sys.modules["anthropic"] = None makes `import anthropic` raise
            # ImportError — the exact condition that used to crash the header.
            out = bc.ask_vision("what's on screen?", b"pngbytes")
        self.assertEqual(out, "[local-vision] a terminal",
                         "a missing anthropic SDK must degrade to the local VLM")

    def test_missing_sdk_and_no_local_returns_honest_string(self):
        bc = self.bc
        with mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch("core.config.model_route", return_value="auto"), \
             mock.patch.object(bc, "_call_local_vision", return_value=None), \
             mock.patch.dict(sys.modules, {"anthropic": None}):
            out = bc.ask_vision("q", b"png")
        self.assertIsInstance(out, str)
        self.assertIn("anthropic SDK", out)   # honest, not an exception


@requires_monolith
class KinectBoundedAcquireTests(unittest.TestCase):
    """Finding #32. _open_runtime_locked used a plain `with _open_attempt_lock:`.
    The locked body runs a retry gauntlet (up to ~16s on a wedged sensor), and
    the negative cache that would fast-fail a second caller isn't published until
    that gauntlet finishes — so the VOICE THREAD's get_runtime() blocked for the
    whole gauntlet. The acquire must be bounded so a second caller fails fast."""

    def setUp(self):
        from audio import kinect_bridge as kb
        self.kb = kb
        self._orig_enabled = kb._ENABLED
        self._orig_rt = kb._runtime[0]
        self._orig_err = kb._open_error[0]
        kb._ENABLED = True
        kb._runtime[0] = None
        kb._open_error[0] = None

    def tearDown(self):
        kb = self.kb
        kb._ENABLED = self._orig_enabled
        kb._runtime[0] = self._orig_rt
        kb._open_error[0] = self._orig_err

    def test_second_caller_fails_fast_when_lock_held(self):
        import threading, time as _t
        kb = self.kb
        held = threading.Event()
        release = threading.Event()

        def _hog():
            with kb._open_attempt_lock:
                held.set()
                release.wait(timeout=5)

        hog = threading.Thread(target=_hog, daemon=True)
        hog.start()
        self.assertTrue(held.wait(timeout=2), "hog must own the lock")
        try:
            t0 = _t.monotonic()
            rt, err = kb._open_runtime_locked()
            dt = _t.monotonic() - t0
        finally:
            release.set()
        # THE observable that pins the fix: it returns FAST while the lock is
        # held, instead of blocking on the up-to-16s open gauntlet (old bug) or
        # until the hog releases at 5s. It returns either an honest "in progress"
        # error (contended, nothing cached) or a cached runtime — never a hang.
        self.assertLess(dt, 2.0,
                        "a second caller must fail fast (~0.5s), not block on "
                        "the open gauntlet")
        if rt is None:
            self.assertIsNotNone(err, "no runtime → must give an honest reason")


class SelfDiagnosticCameraWakeBoundedTests(unittest.TestCase):
    """Finding #44. _do_wake's `io_lock.acquire()` had no timeout. Only the
    caller's join was bounded, so the worker blocked forever waiting for the
    tracker's lock — and when it finally won, LONG after the caller gave up, it
    opened the camera anyway, stealing the device. The worker must bound its own
    wait and touch nothing if it can't get the lock in time."""

    def test_worker_reports_contention_instead_of_stealing_the_device(self):
        import threading
        import types as _types
        import skills.self_diagnostic as sd

        lock = threading.Lock()
        lock.acquire()   # simulate the face-tracker holding the camera
        fake_bc = mock.Mock()
        fake_bc._camera_io_lock = lock
        # A minimal fake cv2 so the function gets past its own import; its
        # VideoCapture must NEVER be called when the lock is contended.
        fake_cv2 = _types.SimpleNamespace(
            CAP_DSHOW=0,
            VideoCapture=mock.Mock(side_effect=AssertionError(
                "device opened despite the lock being held — the exact "
                "device-steal this fix prevents")))
        try:
            with mock.patch.object(sd, "_bc", return_value=fake_bc), \
                 mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
                ok, note = sd._attempt_camera_wake(3, timeout_s=0.4)
        finally:
            lock.release()
        self.assertFalse(ok)
        self.assertIn("lock held", note.lower(),
                      "a contended wake must report contention, not open the cam")


class EnrollVoiceBoundedWaitTests(unittest.TestCase):
    """Finding #37. Both enrollment recorders called a bare sd.wait() on the
    voice thread — a mic that opens but never streams blocks it FOREVER. The
    audit cited skills/enroll_voice.py; a grep-every-copy pass (this codebase's
    #1 lesson) found the same unbounded wait in tools/enroll_voice.py."""

    def test_tools_recorder_times_out_on_a_stalled_mic(self):
        import threading
        import types as _types
        import numpy as np
        from tools import enroll_voice as ev

        fake_sd = _types.SimpleNamespace(
            rec=lambda *a, **k: np.zeros((10, 1), dtype="float32"),
            wait=lambda: threading.Event().wait(),   # never returns — stalled mic
            stop=lambda: None,
        )
        # soundfile is imported at the top of _record_reference_wav and is
        # BLOCKED in the CI-sim tier; fake it so the test exercises the wait, not
        # the import. (It must not be reached anyway — the TimeoutError fires
        # first — but the import itself would raise before we get there.)
        fake_sf = _types.SimpleNamespace(write=lambda *a, **k: None)
        done = {}

        def _run():
            try:
                ev._record_reference_wav("unit-test-profile", 0.1)
            except BaseException as e:      # noqa: BLE001 — capture for assert
                done["exc"] = e

        with mock.patch.dict(sys.modules,
                             {"sounddevice": fake_sd, "soundfile": fake_sf}):
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=6)
        self.assertFalse(t.is_alive(),
                         "a stalled mic must NOT block enrolment forever")
        self.assertIsInstance(done.get("exc"), TimeoutError)

    def test_neither_copy_has_an_unbounded_bare_wait(self):
        import re
        for rel in ("skills/enroll_voice.py", "tools/enroll_voice.py"):
            with open(os.path.join(_PROJECT, rel), encoding="utf-8") as fh:
                lines = fh.readlines()
            # Every sd.wait() must sit inside a bounded wrapper: the same file
            # must also contain a `_done`/`_await` Event bound near it. Cheap
            # structural guard against a future copy regressing.
            has_wait = any(re.search(r"\bsd\.wait\(\)", ln) for ln in lines)
            has_bound = any("wait(timeout=" in ln for ln in lines)
            self.assertTrue(has_wait, f"{rel}: expected an sd.wait somewhere")
            self.assertTrue(has_bound,
                            f"{rel}: sd.wait must be guarded by a bounded "
                            f"Event.wait(timeout=...) — unbounded wait regressed")


class ScreenshotVerifiesBeforeClaimingTests(unittest.TestCase):
    """Findings #40/#43 (same defect). _act_screenshot's PowerShell fallback
    returned "screenshot saved" UNCONDITIONALLY — capture_output threw stderr
    away, check= wasn't passed, the CompletedProcess was discarded. So a
    PowerShell error (assembly load failure, locked disk, bad path) reported
    success while no file existed, and a downstream vision step then read a
    stale/absent image."""

    def _fake_bc(self, tmp):
        bc = mock.Mock()
        bc.__file__ = os.path.join(tmp, "bobert_companion.py")
        bc.screenshot_privacy_block_reason.return_value = None
        bc.take_screenshot.return_value = None   # force the PowerShell fallback
        return bc

    def test_powershell_failure_is_reported_not_claimed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bc = self._fake_bc(tmp)
            fail = mock.Mock(returncode=1, stderr=b"Save failed: path locked")
            with mock.patch.object(A, "_bc", return_value=bc), \
                 mock.patch.object(sys, "platform", "win32"), \
                 mock.patch.object(A.subprocess, "run", return_value=fail):
                out = A._act_screenshot("")
        low = out.lower()
        self.assertTrue("fail" in low or "error" in low,
                        f"a failed screenshot must not claim success: {out!r}")
        self.assertNotIn("screenshot saved", low)

    def test_powershell_success_requires_a_real_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bc = self._fake_bc(tmp)
            ok = mock.Mock(returncode=0, stderr=b"")
            # returncode 0 but NO file on disk → still a failure, not "saved".
            with mock.patch.object(A, "_bc", return_value=bc), \
                 mock.patch.object(sys, "platform", "win32"), \
                 mock.patch.object(A.subprocess, "run", return_value=ok), \
                 mock.patch.object(A.os.path, "exists", return_value=False):
                out = A._act_screenshot("")
        self.assertNotIn("screenshot saved", out.lower(),
                         "exit 0 with no file on disk is still a failure")

    def test_powershell_real_success_reports_saved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bc = self._fake_bc(tmp)
            ok = mock.Mock(returncode=0, stderr=b"")
            with mock.patch.object(A, "_bc", return_value=bc), \
                 mock.patch.object(sys, "platform", "win32"), \
                 mock.patch.object(A.subprocess, "run", return_value=ok), \
                 mock.patch.object(A.os.path, "exists", return_value=True), \
                 mock.patch.object(A.os.path, "getsize", return_value=4096):
                out = A._act_screenshot("")
        self.assertIn("screenshot saved", out.lower())


class HudCardTemperatureIsFahrenheitTests(unittest.TestCase):
    """Finding #33. hud_card stored and rendered raw Celsius while every other
    surface (unified HUD) converts to Fahrenheit — the transient card showed
    "21°C" beside the HUD's "70°F" for the same reading, unusable for a US owner
    (this project's units rule)."""

    def test_current_weather_carries_fahrenheit(self):
        import hud_card
        fake = {"current_condition": [
            {"temp_C": "21", "weatherDesc": [{"value": "Sunny"}]}]}
        with mock.patch.object(hud_card, "_fetch_wttr", return_value=fake):
            w = hud_card._gather_weather_now()
        self.assertIsNotNone(w)
        self.assertEqual(w["temp_f"], round(21 * 9 / 5 + 32))   # 70
        self.assertEqual(w["temp_f"], 70)

    def test_forecast_carries_fahrenheit(self):
        import hud_card
        fake = {"weather": [
            {"maxtempC": "25", "mintempC": "10", "date": "2026-07-14",
             "hourly": [{"time": "1200", "weatherDesc": [{"value": "Clear"}]}]}]}
        with mock.patch.object(hud_card, "_fetch_wttr", return_value=fake):
            days = hud_card._gather_forecast()
        self.assertTrue(days)
        self.assertEqual(days[0]["high_f"], 77)   # 25C
        self.assertEqual(days[0]["low_f"], 50)    # 10C

    def test_renderer_uses_fahrenheit_keys(self):
        with open(os.path.join(_PROJECT, "hud_card.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("temp_f", src)
        self.assertNotIn("temp_c', '?')}°C", src,
                         "the big headline number must not render raw Celsius")


class TeamsNoDeclineAfterOutOfBandAnswerTests(unittest.TestCase):
    """Finding #31. The auto-decline grace thread gated only on the in-memory
    armed flag, which is cleared ONLY by JARVIS's own voice actions. If the user
    answered in the Teams UI (or the caller hung up) the flag stayed armed, and
    at t+grace the thread fired Ctrl+Shift+D into a call the user had already
    taken, plus a focus steal. It must re-check live window state first."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("teams_screener")
        self.mod.PRIORITY_AUTO_DECLINE_SECONDS = 0    # no real wait in the test

    def test_grace_thread_does_not_decline_an_ended_call(self):
        import time as _t
        mod = self.mod
        vip = {"name": "Wayne", "priority": True}
        sent = mock.Mock(return_value=True)
        with mock.patch.object(mod, "_send_teams_hotkey", sent), \
             mock.patch.object(mod, "_pause_music_via_main", return_value=True), \
             mock.patch.object(mod, "_enqueue_speech"), \
             mock.patch.object(mod, "_detect", return_value=("none", None, "")):
            mod._arm_call(vip)
            mod._vip_priority_handler(vip)     # spawns the grace thread
            _t.sleep(0.5)                       # let it run (grace=0)
        sent.assert_not_called()               # NO decline hotkey fired
        self.assertIsNone(mod._active_call,
                          "an ended call must clear our arming, not decline")

    def test_grace_thread_still_declines_a_genuinely_unanswered_call(self):
        import time as _t
        mod = self.mod
        vip = {"name": "Wayne", "priority": True}
        sent = mock.Mock(return_value=True)
        with mock.patch.object(mod, "_send_teams_hotkey", sent), \
             mock.patch.object(mod, "_pause_music_via_main", return_value=True), \
             mock.patch.object(mod, "_enqueue_speech"), \
             mock.patch.object(mod, "_detect",
                               return_value=("call", vip, "Wayne | Microsoft Teams")):
            mod._arm_call(vip)
            mod._vip_priority_handler(vip)
            _t.sleep(0.5)
        sent.assert_called()   # still ringing after grace → decline as designed


@requires_monolith
class ApplePlaylistMediaWindowContractTests(unittest.TestCase):
    """Finding #25. _apple_music_play_playlist opened Library>Playlists with a
    bare _open_url_in_browser — no close_hwnd, and it never recorded the new
    window. So the prior track's window lingered (its still-playing title could
    satisfy the confirm scan → false "playing") and the fullscreen key hit the
    stale hwnd. It must use the same media-window contract as the search flow."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_playlist_open_closes_prior_and_records_new_hwnd(self):
        bc = self.bc
        opened = {}

        def _fake_open(url, close_matching=None, close_hwnd=None):
            opened["url"] = url
            opened["close_hwnd"] = close_hwnd
            return "chrome"

        fake_win = mock.Mock()
        fake_win._hWnd = 4242
        bc._JARVIS_MEDIA_WINDOW_HWND["apple_music"] = 111   # a prior window

        with mock.patch.object(bc, "_open_url_in_browser", side_effect=_fake_open), \
             mock.patch.object(bc, "_find_browser_window_matching", return_value=fake_win), \
             mock.patch.object(bc, "_ensure_window_visible_maximized", return_value=True), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", False), \
             mock.patch("time.sleep"):
            # SCREEN_VISION_ENABLED False makes it return right after the open +
            # record block — which is all this test needs to observe.
            bc._apple_music_play_playlist("workout")

        self.assertEqual(opened.get("close_hwnd"), 111,
                         "must close the window JARVIS opened last time")
        self.assertEqual(bc._JARVIS_MEDIA_WINDOW_HWND.get("apple_music"), 4242,
                         "must record the freshly-opened window's hwnd so the "
                         "confirm + fullscreen target it, not a stale sibling")


class CheckNoPiiImportIsSafeTests(unittest.TestCase):
    """Finding #41. check_no_pii ran _load_local_patterns() at MODULE scope, and
    that loader could sys.exit(2) on a degraded pii_local.py — so merely
    `import tools.check_no_pii` (as core/bug_reporter does to scrub by the same
    rules) could TERMINATE the interpreter, violating bug_reporter's "never
    crash the host" contract. The gate must still fail closed; importing must
    not exit."""

    def setUp(self):
        import tools.check_no_pii as cnp
        self.cnp = cnp
        self._orig_err = cnp._LOAD_ERROR
        self._orig_cands = cnp._local_pattern_candidates

    def tearDown(self):
        self.cnp._LOAD_ERROR = self._orig_err
        self.cnp._local_pattern_candidates = self._orig_cands

    def test_degraded_local_file_does_not_exit_the_interpreter(self):
        import tempfile
        cnp = self.cnp
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "pii_local.py")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("this is not valid python !!! def (\n")
            cnp._LOAD_ERROR = None
            cnp._local_pattern_candidates = lambda: [bad]
            try:
                cnp._load_local_patterns()   # must NOT raise SystemExit
            except SystemExit as e:           # pragma: no cover - the bug
                self.fail(f"importing/loading exited the interpreter: {e}")
        self.assertIsNotNone(cnp._LOAD_ERROR,
                             "a broken pii_local.py must be recorded as degraded")

    def test_gate_still_fails_closed_when_degraded(self):
        cnp = self.cnp
        cnp._LOAD_ERROR = "simulated degraded scanner"
        try:
            rc = cnp.main([])
        except SystemExit as e:               # pragma: no cover
            self.fail(f"main() should RETURN 2, not exit: {e}")
        self.assertEqual(rc, 2, "the git gate must fail closed on a degraded "
                                "scanner (return 2), just not by exiting at import")

    def test_loader_makes_no_real_sys_exit_call(self):
        """The bare `_load_local_patterns()` at module scope stays — it must
        populate HARD/WARN for importers — but it must NOT be able to exit. Use
        AST (not a text grep, which would match this fix's OWN comment that
        discusses sys.exit): assert the loader contains no Call to sys.exit."""
        import ast
        with open(os.path.join(_PROJECT, "tools", "check_no_pii.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        loader = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "_load_local_patterns"), None)
        self.assertIsNotNone(loader)
        exits = [n for n in ast.walk(loader)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "exit"]
        self.assertEqual(exits, [],
                         "the pattern loader must never call sys.exit — it is "
                         "imported by bug_reporter; record _LOAD_ERROR instead")


class CiSimBlocksWinOnlyFromlistTests(unittest.TestCase):
    """Finding #36. The ci-sim import shim inspected only the module NAME. `from
    ctypes import wintypes` calls __import__("ctypes", ..., ["wintypes"]) — the
    package `ctypes` is importable, so the win-only child `ctypes.wintypes`
    leaked through, and `from ctypes import wintypes` at module top passed the
    sim while it would ModuleNotFoundError on the real Linux runner."""

    def test_blocked_recognises_the_winonly_submodule(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_ci_sim_probe",
            os.path.join(_PROJECT, "tools", "run_tests_ci_sim.py"))
        mod = importlib.util.module_from_spec(spec)
        # Importing run_tests_ci_sim runs only defs at module top (its work is
        # under `if __name__ == "__main__"`), so this is cheap and side-effect
        # free — we just want its _blocked predicate.
        spec.loader.exec_module(mod)
        self.assertTrue(mod._blocked("ctypes.wintypes"),
                        "ctypes.wintypes is win-only and must be blocked")
        self.assertFalse(mod._blocked("ctypes"),
                         "plain ctypes is Linux-safe and must NOT be blocked — "
                         "so the fromlist check is the ONLY thing that can catch "
                         "the submodule")

    def test_imp_inspects_the_fromlist(self):
        """Structural guard: the shim's real-import hook must examine fromlist,
        otherwise a win-only child re-leaks through the package head."""
        with open(os.path.join(_PROJECT, "tools", "run_tests_ci_sim.py"),
                  encoding="utf-8") as fh:
            body = fh.read()
        imp = body.split("def _imp(")[1].split("\n    def ")[0]
        self.assertIn("fromlist", imp,
                      "_imp must inspect fromlist to block win-only submodules")


class GenActionIndexCoversPackagesAndAliasesTests(unittest.TestCase):
    """Findings #34 + #45. gen_action_index globbed only skills/*.py (missing
    PACKAGE skills' __init__.py → whole packages rendered with `?`), and its
    RHS capture stopped at `[`, so `actions["a"] = actions["b"]` recorded the
    bare symbol `actions` (a non-existent handler → `?`)."""

    def _src(self):
        with open(os.path.join(_PROJECT, "tools", "gen_action_index.py"),
                  encoding="utf-8") as fh:
            return fh.read()

    def test_globs_include_package_inits(self):
        src = self._src()
        self.assertIn('"skills", "*", "__init__.py"', src,
                      "must glob package skills' __init__.py, not only *.py")

    def test_alias_form_regex_captures_both_names(self):
        import re
        # The exact alias-resolution pattern the fix added; prove it captures
        # (alias, target) from a real registration line, where the old single
        # regex captured RHS == "actions".
        pat = re.compile(
            r'actions\[\s*[\'"]([a-zA-Z_0-9]+)[\'"]\s*\]\s*=\s*'
            r'actions\[\s*[\'"]([a-zA-Z_0-9]+)[\'"]\s*\]')
        m = pat.search('    actions["schedule_cron"] = actions["schedule_recurring"]')
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "schedule_cron")
        self.assertEqual(m.group(2), "schedule_recurring")

    def test_generated_index_resolves_a_known_alias(self):
        """End-to-end: the committed ACTION_INDEX (regenerated by this fix) must
        show schedule_cron sharing schedule_recurring's real location, not `?`.
        The alias source lives in skills/schedule_manager.py."""
        idx_path = os.path.join(_PROJECT, "docs", "ACTION_INDEX.md")
        if not os.path.exists(idx_path):
            self.skipTest("ACTION_INDEX.md not present in this checkout")
        with open(idx_path, encoding="utf-8") as fh:
            idx = fh.read()
        # Find the row mentioning schedule_cron and assert it carries a real
        # schedule_manager location rather than a bare '?'.
        rows = [ln for ln in idx.splitlines() if "schedule_cron" in ln]
        self.assertTrue(rows, "schedule_cron should appear in the index")
        self.assertTrue(any("schedule_manager.py:" in ln for ln in rows),
                        f"schedule_cron alias must resolve to a real location, "
                        f"not '?': {rows}")


# ═════════════════════════════════════════════════════════════════════════
#  FRESH BUG-HUNT (2026-07-14, post all-45-closed) — confirmed NEW defects
# ═════════════════════════════════════════════════════════════════════════

class TelegramWhitelistFailsClosedTests(unittest.TestCase):
    """Bug-hunt HIGH #2 + green-by-mock #16. An empty TELEGRAM_USER_ID whitelist
    means "deny all" (per _telegram_whitelist's docstring). But _process_update
    guarded with `if whitelist and ...`, so an empty set short-circuited to False
    and admitted EVERYONE straight into _dispatch_remote — arbitrary JARVIS
    actions from any Telegram user. A unit test even PINNED the bypass as
    correct. The whitelist is the only inbound auth boundary; it must fail
    closed."""

    def setUp(self):
        self.mod, _actions = load_skill_isolated("phone_bridge")

    def _update(self, uid, cid, text):
        return {"update_id": 1, "message": {
            "text": text, "chat": {"id": cid}, "from": {"id": uid}}}

    def test_empty_whitelist_rejects_and_does_not_dispatch(self):
        mod = self.mod
        with mock.patch.object(mod, "_dispatch_remote") as disp, \
             mock.patch.object(mod, "_send_telegram", return_value=True):
            mod._process_update(self._update(5, 5, "turn on the lights"),
                                whitelist=set())
        disp.assert_not_called()

    def test_whitelisted_user_still_dispatches(self):
        mod = self.mod
        with mock.patch.object(mod, "_dispatch_remote",
                               return_value="done") as disp, \
             mock.patch.object(mod, "_send_telegram", return_value=True):
            mod._process_update(self._update(42, 42, "status"), whitelist={42})
        disp.assert_called_once()

    def test_unwhitelisted_user_with_nonempty_list_rejected(self):
        mod = self.mod
        with mock.patch.object(mod, "_dispatch_remote") as disp, \
             mock.patch.object(mod, "_send_telegram", return_value=True):
            mod._process_update(self._update(999, 999, "hi"), whitelist={42})
        disp.assert_not_called()


class KinectWedgedCooldownSharedTests(unittest.TestCase):
    """Bug-hunt HIGH #1. available() and get_runtime() share the negative-cache
    cells, but available() ALWAYS wrote the short 5s cooldown while get_runtime()
    correctly applied the 90s WEDGED cooldown for a no-frames open. So an
    available() call (fired ~30Hz by the air-mouse/two-hand pollers and on the
    voice thread by kinect_status) re-armed the ~16s open gauntlet every few
    seconds, defeating the cooldown that keeps a wedged sensor off the voice
    loop. Both now go through one shared _publish_open_failure."""

    def setUp(self):
        from audio import kinect_bridge as kb
        self.kb = kb
        self._enabled = kb._ENABLED
        self._rt = kb._runtime[0]
        self._nu = kb._negative_until[0]
        self._oe = kb._open_error[0]
        kb._ENABLED = True
        kb._runtime[0] = None
        kb._negative_until[0] = 0.0
        kb._open_error[0] = None

    def tearDown(self):
        kb = self.kb
        kb._ENABLED = self._enabled
        kb._runtime[0] = self._rt
        kb._negative_until[0] = self._nu
        kb._open_error[0] = self._oe

    def test_available_uses_wedged_cooldown_for_no_frames(self):
        import time as _t
        kb = self.kb
        wedged = (None, "Kinect opened but streamed no frames after 4 attempts")
        with mock.patch.object(kb, "_open_runtime_locked", return_value=wedged):
            ok, _err = kb.available()
        self.assertFalse(ok)
        remaining = kb._negative_until[0] - _t.monotonic()
        # Must be near the 90s wedged cooldown, NOT the 5s short one.
        self.assertGreater(remaining, kb._NEGATIVE_CACHE_SEC + 5,
                           "available() must honour the long wedged cooldown so "
                           "it can't stomp get_runtime()'s 90s value down to 5s")
        self.assertLessEqual(remaining, kb._WEDGED_CACHE_SEC + 1)

    def test_available_uses_short_cooldown_for_plain_failure(self):
        import time as _t
        kb = self.kb
        with mock.patch.object(kb, "_open_runtime_locked",
                               return_value=(None, "could not open Kinect sensor")):
            kb.available()
        remaining = kb._negative_until[0] - _t.monotonic()
        self.assertLessEqual(remaining, kb._NEGATIVE_CACHE_SEC + 1)

    def test_disabled_earns_no_cooldown(self):
        kb = self.kb
        kb._negative_until[0] = 0.0
        with mock.patch.object(kb, "_open_runtime_locked",
                               return_value=(None, "Kinect is disabled")):
            kb.available()
        self.assertEqual(kb._negative_until[0], 0.0,
                         "disabled-by-config is cheap to re-check; no cooldown")


class WhichMonitorUsesCanonicalSideTests(unittest.TestCase):
    """Bug-hunt #9. _act_which_monitor used a raw `look_x < 0.5` — the 4th copy
    of the rule the other three call sites migrated off. It ignores the camera
    LABEL and sends look_x==0.5 (the live LEFT cam's value) to the wrong side.
    Must use the monolith's canonical _percam_side (label-first, <=0.5)."""

    def _percam_side(self, cam):
        # Faithful copy of bobert_companion._percam_side (label-first, <=0.5)
        # so the test needs no heavy monolith import.
        lbl = str(cam.get("label", "")).lower()
        if "left" in lbl:
            return "left"
        if "right" in lbl:
            return "right"
        return "left" if cam.get("look_x", 0.5) <= 0.5 else "right"

    def test_label_wins_over_look_x(self):
        import threading
        import types as _types
        cams = [{"index": 2, "label": "Left webcam (left monitor)", "look_x": 0.9},
                {"index": 0, "label": "Right webcam", "look_x": 0.1}]
        bc = mock.Mock()
        bc._percam_side = self._percam_side
        bc._camera_state_lock = threading.Lock()
        bc._camera_last_seen = {2: 9e18, 0: 0.0}   # only index 2 is visible
        bc._camera_latest_frame = {2: None}
        fake_cv2 = _types.SimpleNamespace()
        with mock.patch.object(A, "_bc", return_value=bc), \
             mock.patch.object(A, "_kinect_gaze_which_monitor", return_value=None), \
             mock.patch("core.config.CAMERAS", cams), \
             mock.patch("core.config.MONITORS", ["left", "right", "top"]), \
             mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            out = A._act_which_monitor("")
        # Visible only to index 2, whose LABEL says left — despite look_x=0.9
        # which the old raw `look_x < 0.5` rule would have called "right".
        self.assertIn("LEFT", out.upper(),
                      f"label-first side rule must win over look_x: {out!r}")


@requires_monolith
class OllamaBranchUsesInstrumentedFallbackTests(unittest.TestCase):
    """Bug-hunt #6. _call_llm's ollama branch degraded to a dead-end string on
    failure, while the claude branch and the two sibling call sites route
    through _local_fallback_or → _call_local_llm, which carries the ollama
    SELF-HEAL + wedge-reap machinery. So a wedged runner just said "Ollama may
    be down" instead of being restarted and retried."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_ollama_failure_routes_through_local_fallback(self):
        bc = self.bc
        # model_route("chat")=="local" is the FIRST branch of _call_llm (a
        # separate _call_local_llm path); force it off "local" so the
        # AI_BACKEND=="ollama" branch — the one this fix touches — is reached.
        with mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch("core.config.model_route", return_value="auto"), \
             mock.patch.object(bc, "detect_tone", return_value=None), \
             mock.patch.object(bc, "route_voice_emotion",
                               return_value={"mood": "casual", "addendum": ""}), \
             mock.patch.object(bc, "_ltm_enqueue"), \
             mock.patch.object(bc, "_ollama_chat_bounded",
                               side_effect=RuntimeError("wedged")), \
             mock.patch.object(bc, "_local_fallback_or",
                               return_value="recovered locally") as fb, \
             mock.patch.object(bc, "conversation_history", []):
            out = bc._call_llm("hi")
        fb.assert_called_once()   # the self-healing path, not a dead-end string
        self.assertEqual(out, "recovered locally")


class SwitchLlmTagIsAuthoritativeTests(unittest.TestCase):
    """Bug-hunt #7. _act_switch_llm set bc.OLLAMA_MODEL = tag, but every
    generation resolves through _get_local_llm_model() (the
    _RESOLVED_LOCAL_LLM_MODEL cache), which never reads OLLAMA_MODEL — so "switch
    to qwen2.5:14b" reported success while the box kept answering on gemma4.
    The pick must repoint the resolver cache (only when the tag is installed)."""

    def _bc(self, installed):
        bc = mock.Mock()
        bc.AI_BACKEND = "claude"
        bc._KNOWN_OLLAMA_MODELS = {"qwen2.5:14b"}
        bc._RESOLVED_LOCAL_LLM_MODEL = ["gemma4:12b"]
        bc.OLLAMA_MODEL = "gemma4:12b"
        bc.LOCAL_LLM_MODEL = "gemma4:12b"
        bc._ollama_has_model.return_value = installed
        bc._get_local_llm_model.return_value = "gemma4:12b"
        return bc

    def test_installed_tag_repoints_the_resolver_cache(self):
        bc = self._bc(installed=True)
        with mock.patch.object(A, "_bc", return_value=bc):
            out = A._act_switch_llm("qwen2.5:14b")
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], "qwen2.5:14b",
                         "an installed pick must take effect next turn")
        self.assertIn("qwen2.5:14b", out)

    def test_uninstalled_tag_pulls_and_keeps_current_model(self):
        bc = self._bc(installed=False)
        with mock.patch.object(A, "_bc", return_value=bc):
            out = A._act_switch_llm("qwen2.5:14b")
        # Must NOT repoint the cache to an uninstalled tag (would 404 every turn).
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], "gemma4:12b")
        bc._ollama_pull_async.assert_called_once_with("qwen2.5:14b")
        self.assertIn("isn't installed", out)


@requires_monolith
class ThinkingLabelShowsResolvedModelTests(unittest.TestCase):
    """Bug-hunt #19. The THINKING label and the tray llm_backend publish read the
    vestigial OLLAMA_MODEL constant (retired "llama3"), not the resolved tag —
    so the HUD said "THINKING (llama3)" while gemma4:12b was on the wire. Read
    the resolver CACHE (never _get_local_llm_model, which HTTP-blocks on the hot
    label path)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_thinking_label_uses_resolved_cache_not_ollama_model(self):
        bc = self.bc
        with mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "OLLAMA_MODEL", "llama3"), \
             mock.patch.object(bc, "_RESOLVED_LOCAL_LLM_MODEL", ["gemma4:12b"]), \
             mock.patch.object(bc, "_get_local_llm_model",
                               side_effect=AssertionError("must not HTTP here")):
            label = bc._now_doing_label("thinking")
        self.assertIn("gemma4:12b", label)
        self.assertNotIn("llama3", label)


@requires_monolith
class StreamingVerifyPinsMonitorTests(unittest.TestCase):
    """Bug-hunt #13. _streaming_verify_playback took a whole-desktop screenshot
    (monitors[0]) while the click path captured just the player's monitor — so
    on multi-monitor the verify vision false-negatived and the last-resort
    media-playpause PAUSED the autoplaying video. The verify capture must use
    the same monitor pin the click path does."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_verify_passes_the_monitor_to_take_screenshot(self):
        bc = self.bc
        seen = {}
        with mock.patch.object(bc, "take_screenshot",
                               side_effect=lambda m=None: seen.setdefault("mon", m) or b"png"), \
             mock.patch.object(bc, "ask_vision", return_value="YES it is playing"), \
             mock.patch.object(bc, "_vision_answer_is_yes", return_value=True):
            bc._streaming_verify_playback("playing?", monitor="right")
        self.assertEqual(seen.get("mon"), "right",
                         "verify must capture the SAME monitor the click path used")

    def test_confirm_threads_cfg_vision_monitor(self):
        bc = self.bc
        seen = {}
        cfg = {"title_confirm": False, "vision_monitor": "left",
               "verify_question": "is it playing?", "service_key": "youtube"}

        def _fake_verify(q, monitor=None):
            seen["mon"] = monitor
            return (True, "YES")

        with mock.patch.object(bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(bc, "_vision_click_backend_available", return_value=True), \
             mock.patch.object(bc, "_streaming_title_confirms_playback",
                               return_value=(False, "no title")), \
             mock.patch.object(bc, "_streaming_verify_playback",
                               side_effect=_fake_verify):
            bc._streaming_confirm_playback(cfg, "is it playing?")
        self.assertEqual(seen.get("mon"), "left")


@requires_monolith
class AppleMusicConfirmScopedToWindowTests(unittest.TestCase):
    """Bug-hunt #14. _apple_music_title_now_playing scanned EVERY browser window,
    so a user's own second Apple Music tab (still showing an earlier track)
    false-confirmed — JARVIS reported "playing <requested>" while the requested
    track never started. The confirm path must scope to the recorded media
    window's hwnd."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _win(self, hwnd, title):
        w = mock.Mock()
        w._hWnd = hwnd
        w.title = title
        return w

    def test_only_hwnd_ignores_other_windows(self):
        bc = self.bc
        import types as _types
        # A stale OTHER Apple Music tab (hwnd 111) plus our window (hwnd 222,
        # no track yet). Scoped to 222, the stale 111 must be ignored.
        wins = [self._win(111, "Old Song — Old Artist - Google Chrome"),
                self._win(222, "Apple Music - Google Chrome")]
        fake_gw = _types.SimpleNamespace(getAllWindows=lambda: wins)
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw}):
            scoped = bc._apple_music_title_now_playing(only_hwnd=222)
            unscoped = bc._apple_music_title_now_playing()
        self.assertIsNone(scoped,
                          "a stale OTHER Apple Music tab must not confirm our request")
        # Unscoped still sees the stale tab (the now_playing action wants any tab).
        self.assertIsNotNone(unscoped)


class YoutubeDirectDoesNotOverclaimTests(unittest.TestCase):
    """Bug-hunt #22. youtube_search_direct did NO playback verification yet
    claimed "video is now playing, no further action needed" — false success on
    an unavailable/age-gated video, and the phrasing suppressed correction."""

    def test_no_false_playing_claim(self):
        with open(os.path.join(_PROJECT, "skills", "youtube_search.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("video is now playing, no further action needed", src)
        self.assertNotIn("no further action needed", src)


@requires_monolith
class StandbyWakeGateWordBoundaryTests(unittest.TestCase):
    """Bug-hunt #5. The standby/sleep wake gate used a raw substring test
    (`any(wp in tl ...)`), the stale duplicate of the word-boundary matcher the
    ambient listener already used — so "awakened" matched "wake" and "jar visit"
    matched "jarvis"."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_substring_false_positives_are_rejected(self):
        bc = self.bc
        for bad in ("the machine awakened slowly", "he paid a jar visit",
                    "i awakened early"):
            self.assertIsNone(bc._WAKE_RE.search(bad),
                              f"{bad!r} must NOT trip the wake gate")

    def test_real_wake_phrases_match(self):
        bc = self.bc
        for good in ("jarvis", "hey jarvis are you there", "wake up",
                     "jarvis!", "please wake up"):
            self.assertIsNotNone(bc._WAKE_RE.search(good),
                                 f"{good!r} must trip the wake gate")


@requires_monolith
class SpeakReportsPlaybackOutcomeTests(unittest.TestCase):
    """Bug-hunt #18. _speak swallowed playback failures (by design) but returned
    nothing, so the streaming flush ledger recorded a TTS-FAILED sentence as
    spoken — and _strip_stream_spoken_prefix then dropped it from the tail. _speak
    now returns True on success / False on failure; the ledger records only when
    it's not False."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_speak_returns_false_on_playback_failure(self):
        bc = self.bc
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "synthesise", return_value=(b"a", 22050)), \
             mock.patch.object(bc, "play_with_lipsync",
                               side_effect=RuntimeError("device gone")), \
             mock.patch.object(bc, "set_state"):
            out = bc._speak("hello sir")
        self.assertIs(out, False)

    def test_speak_returns_true_on_success(self):
        bc = self.bc
        with mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "synthesise", return_value=(b"a", 22050)), \
             mock.patch.object(bc, "play_with_lipsync"), \
             mock.patch.object(bc, "set_state"):
            out = bc._speak("hello sir")
        self.assertIs(out, True)


class CameraProbeManyBudgetTests(unittest.TestCase):
    """Bug-hunt #3. _probe_many joined each runner with a flat 3.5s timeout, but
    _probe_camera_index internally allows ~42s for the camera-lock QUEUE (a dead
    index holds the lock 20-30s per cv2.VideoCapture). So a runner abandoned at
    3.5s never wrote results[i] and a PRESENT camera queued behind a ghost was
    dropped. The batch join must be budget-aware."""

    def test_batch_join_is_budget_aware_not_flat_per_thread(self):
        # The _probe_many join lives in a nested closure inside
        # probe_cameras_and_update_config, so assert the contract at the source
        # level: the join budget must scale with batch size (mirroring the ~42s
        # PHASE-1 lock_budget), and the old flat per-thread cap must be gone.
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("(CAMERA_PROBE_TIMEOUT_SEC + 0.5) * max(1, len(indices))", src,
                      "the batch join must scale with batch size, not use a flat "
                      "per-thread cap that abandons probes queued on the lock")
        self.assertNotIn("t.join(timeout=CAMERA_PROBE_TIMEOUT_SEC + 0.5)", src,
                         "the old flat per-thread cap must be gone")


@requires_monolith
class DevicePeriodicReenumTests(unittest.TestCase):
    """Bug-hunt #4. The device signature is sourced from sd.query_devices(),
    which PortAudio freezes until the reinit the signature gate blocks — so USB
    hotplug never trips it and mic auto-switch was dead in steady state. Fix
    keeps the cheap 4s gate but forces a real re-enumeration every
    DEVICE_REENUM_INTERVAL when idle. (Removing the gate outright would reinit
    every 4s — a 0xc0000374 window every 4s.)"""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_reenum_interval_is_far_longer_than_check_interval(self):
        bc = self.bc
        self.assertGreaterEqual(bc.DEVICE_REENUM_INTERVAL, 60.0,
                                "the forced re-enum must be RARE vs the 4s "
                                "signature check — a 4s destructive teardown "
                                "cadence would reopen the 0xc0000374 window")
        self.assertGreater(bc.DEVICE_REENUM_INTERVAL, bc.DEVICE_CHECK_INTERVAL * 10)

    def test_last_reenum_at_tracked_in_cache(self):
        self.assertIn("last_reenum_at", self.bc._device_cache)


class DispatcherCommaIsGatedTests(unittest.TestCase):
    """Bug-hunt #10. _STRONG_SEP_RE split eagerly on ANY bare comma, tearing a
    comma-bearing ENTITY ("Earth, Wind and Fire") into a bogus second action.
    Bare " and " was already gated on a command-verb RHS; bare comma must be
    gated the same way."""

    def test_entity_commas_are_not_split(self):
        from core.dispatcher import _split_chain
        for u in ("play Earth, Wind and Fire",
                  "set a timer for 1, 2, 3",
                  "remind me to call Bob, Jr."):
            segs = _split_chain(u)
            self.assertEqual(len(segs), 1,
                             f"{u!r} is one action, not a chain: {segs}")

    def test_real_command_boundary_still_splits(self):
        from core.dispatcher import _split_chain
        segs = _split_chain("dim the lights, play jazz")
        self.assertEqual(len(segs), 2, f"a real command boundary must split: {segs}")
        self.assertIn("dim the lights", segs[0])
        self.assertIn("play jazz", segs[1])

    def test_comma_with_connector_is_still_strong(self):
        from core.dispatcher import _split_chain
        # ", and then" is a genuine chain marker (kept in _STRONG_SEP_RE).
        segs = _split_chain("open chrome, and then play music")
        self.assertEqual(len(segs), 2, segs)


@requires_monolith
class OrchestratorSyncPathIsBoundedTests(unittest.TestCase):
    """Bug-hunt #9. orchestrate()'s no-running-loop branch (the path production
    actually takes) called loop.run_until_complete() with NO wall-clock ceiling,
    while the sibling running-loop branch was bounded. A wedged worker could
    block the synchronous main voice turn indefinitely. Both branches must go
    through the same bounded executor."""

    def test_both_paths_use_the_bounded_executor(self):
        # The two branches were unified; assert the direct unbounded
        # run_until_complete on the sync path is gone and the executor+timeout
        # guard is the sole path.
        with open(os.path.join(_PROJECT, "core", "orchestrator.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        orch = src.split("    def orchestrate(")[1].split("\ndef orchestrate(")[0]
        self.assertIn(".result(timeout=self._overall_timeout_s())", orch)
        # Exactly one ACTUAL run_until_complete CALL remains (inside _runner) —
        # count non-comment lines only, so the explanatory comment that mentions
        # run_until_complete() in prose isn't miscounted (a recurring trap).
        calls = [ln for ln in orch.splitlines()
                 if "run_until_complete(" in ln
                 and not ln.lstrip().startswith("#")]
        self.assertEqual(len(calls), 1,
                         f"the sync path must not have its own unbounded "
                         f"run_until_complete: {calls}")


class OrchestratorEmptySummaryKeepsRawDataTests(unittest.TestCase):
    """Bug-hunt #20. _claude_call/_ollama_call return "" (not raise) on an empty
    completion, so an empty summary slipped past the except and dropped the whole
    sub-agent section though real tool data WAS fetched. Fall back to raw data on
    empty-success too."""

    def test_empty_summary_falls_back_to_raw_data(self):
        with open(os.path.join(_PROJECT, "core", "orchestrator.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        worker = src.split("_run_worker_sync")[1]
        self.assertIn('if not (output or "").strip() and (real_data or "").strip():',
                      worker,
                      "an empty summary must fall back to the fetched raw data")


class SmartHomeSceneActivatesTests(unittest.TestCase):
    """Bug-hunt #11. _action_to_kwargs had no branch for verb='scene', so it
    returned {} and _dispatch_one refused every scene with "nothing to do" —
    though the scene verb, discovery type, and "Scene running" summary all
    exist. A scene must emit a kwarg that drives dispatch."""

    def test_scene_verb_emits_a_kwarg(self):
        from core.smart_home_router import _action_to_kwargs
        kw = _action_to_kwargs({"verb": "scene"})
        self.assertTrue(kw, "scene must emit a kwarg so dispatch doesn't refuse")
        self.assertIs(kw.get("on"), True,
                      "an Alexa scene activates by turning its entity ON")

    def test_dispatch_does_not_refuse_a_scene(self):
        from core import smart_home_router as r
        # With a kwarg present, _dispatch_one must get past the
        # "nothing to do" short-circuit. Route it to a stub Alexa path.
        dev = {"name": "Movie", "brand": "alexa", "alexa_entity_id": "e1"}
        with mock.patch.object(r, "_controller_for", return_value=None), \
             mock.patch.object(r, "_log_missing_brand"), \
             mock.patch.object(r, "_alexa_set_state",
                               return_value={"ok": True, "path": "alexa"}) as ax:
            out = r._dispatch_one(dev, {"verb": "scene"})
        ax.assert_called_once()
        self.assertNotIn("nothing to do", str(out))


class AudioProcessorLoopbackStatsTests(unittest.TestCase):
    """Bug-hunt #17. process() wrote the mic-only raw-RMS / rms-history stats
    unconditionally, so the LOOPBACK (system-audio) path polluted the silent-mic
    detector and stress proxy — loud playback made a dead mic look "audible"."""

    def test_process_accepts_record_mic_stats_flag(self):
        import inspect
        from core import audio_processor
        sig = inspect.signature(audio_processor.AudioProcessor.process)
        self.assertIn("record_mic_stats", sig.parameters)
        self.assertIs(sig.parameters["record_mic_stats"].default, True,
                      "default True preserves the real-mic path")

    def test_loopback_caller_passes_false(self):
        with open(os.path.join(_PROJECT, "skills", "ambient_listen.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("record_mic_stats=False", src,
                      "the loopback call must suppress mic-only stats")


class EmailTriageIsBoundedTests(unittest.TestCase):
    """Bug-hunt #16. categorize_inbox/email_briefing ran an unbounded loop of
    un-retry-capped Claude calls on the voice/dispatch thread — up to 50
    messages x (8s x SDK-default-2-retries) = minutes of freeze."""

    def test_clients_cap_retries(self):
        with open(os.path.join(_PROJECT, "skills", "email_triage.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("anthropic.Anthropic()", src,
                         "raw uncapped clients (SDK default max_retries=2) on "
                         "the voice thread must be gone")
        self.assertIn("max_retries=1", src)

    def test_batch_loops_have_a_wall_clock_deadline(self):
        with open(os.path.join(_PROJECT, "skills", "email_triage.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("INBOX_TRIAGE_BUDGET_S", src)
        # Both synchronous batch loops must consult the deadline.
        self.assertGreaterEqual(src.count("time.monotonic() >= _deadline"), 2,
                                "both categorize and briefing loops must bound "
                                "their per-message Claude calls")


class FinalStaleDuplicateSweepTests(unittest.TestCase):
    """Bug-hunt #24/#25/#26/#12/#23 — the tail: mostly the stale-duplicate class
    (a hardened path exists; a sibling copy rotted) plus two cheap
    class-prevention fixes."""

    def test_hud_card_uses_parent_watch_not_psutil(self):
        # #24: hud_card's parent-liveness check must use core.parent_watch (the
        # fleet's canonical, corpse-aware check), not the psutil-backed
        # _pid_alive that reports True for a kernel-stuck corpse.
        with open(os.path.join(_PROJECT, "hud_card.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("from core.parent_watch import parent_is_alive", src)

    def test_keyboardinterrupt_teardown_uses_hardened_release(self):
        # #25: the Ctrl-C teardown must route through _release_native_resources
        # (in-flight wait + close(final=True)), not inlined unload()/close().
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        ki = src.split("except KeyboardInterrupt:")[1][:2000]
        self.assertIn("_release_native_resources(sys.modules[\"__main__\"])", ki)
        # The old inlined stale-duplicate calls must be gone from this block.
        self.assertNotIn("_vc_teardown.unload()", ki)
        self.assertNotIn("_kb_teardown.close()", ki)

    def test_tray_dialog_spawn_and_register_are_atomic(self):
        # #26: Popen must be INSIDE the _dialog_procs_lock with the append, so a
        # concurrent reaper can't orphan a just-spawned dialog.
        with open(os.path.join(_PROJECT, "tray.py"), encoding="utf-8") as fh:
            src = fh.read()
        fn = src.split("def _tracked_dialog_run(")[1].split("\ndef ")[0]
        lock_block = fn.split("with _dialog_procs_lock:")[1]
        self.assertIn("subprocess.Popen(", lock_block.split("_dialog_procs.append")[0],
                      "Popen must happen under the lock, before the append")

    def test_alexa_fallback_runner_is_bounded(self):
        # #12: the alexa runner call must pass a timeout so a stalled endpoint
        # can't wedge the dispatch thread.
        with open(os.path.join(_PROJECT, "core", "smart_home_router.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("_run(_go(), timeout=", src)

    def test_two_hand_overlay_writes_atomically(self):
        # #23: both overlay writes must go through the temp-file+os.replace
        # helper, not a plain open()/json.dump the 60Hz reader can tear.
        with open(os.path.join(_PROJECT, "skills", "kinect_two_hand.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("os.replace(tmp, path)", src)
        pub = src.split("def _publish_two_hand_overlay(")[1]
        self.assertNotIn("json.dump(data, f)", pub,
                         "the publisher must use the atomic writer, not a raw "
                         "json.dump")


if __name__ == "__main__":
    unittest.main()
