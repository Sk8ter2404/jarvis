"""Tests for audio.apple_music_keeper — the opt-in autostart + keep-alive for
the UWP Apple Music app.

The keeper's whole contract is "keep the app open the LEGITIMATE way (launch by
AUMID) and NEVER do it in tests/staging". So everything that could touch the
real app or spawn a thread is faked:

  * audio.apple_music_app is replaced by a fake recording launches / is_running.
  * core.config flags are toggled on the live module and restored.
  * threading.Thread.start is neutered for the idempotency/start tests so no real
    daemon runs; the per-tick logic is asserted via _keep_alive_tick directly.
  * JARVIS_STAGING / a fake monolith _is_staging drive the staging gate.

No real launching, no hardware, no sleeping. stdlib unittest + mock only.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

from audio import apple_music_keeper as K


def _no_keeper_thread_alive() -> bool:
    """True iff NO live OS thread named like the keep-alive watchdog remains."""
    return not any(t.name == K._THREAD_NAME and t.is_alive()
                   for t in threading.enumerate())


def _ensure_no_keeper_thread() -> None:
    """Stop + join any lingering keep-alive watchdog so it can't leak into a
    later test (e.g. the structural compile sweep, which flakes if a stray
    non-terminating daemon survives into it). Belt-and-suspenders: every test
    here already neuters Thread.start, but a regression that lets one slip
    through gets caught and cleaned up right here. Generous join: once _STOP is
    set the loop exits after one wait() wake, ample even under heavy CI load."""
    K.stop_keeper(timeout=10.0)


def tearDownModule():  # noqa: N802 - unittest hook name
    _ensure_no_keeper_thread()


# ─── fakes ──────────────────────────────────────────────────────────────────
class _FakeAppleMusic:
    """Stand-in for audio.apple_music_app. Records launches; never shells out."""
    def __init__(self, running=False, launch_ok=True, launch_err=None,
                 running_raises=False):
        self._running = running
        self.launch_ok = launch_ok
        self.launch_err = launch_err
        self.running_raises = running_raises
        self.launches = 0

    def is_running(self):
        if self.running_raises:
            raise RuntimeError("psutil exploded")
        return self._running

    def launch(self):
        self.launches += 1
        self._running = True
        return self.launch_ok, self.launch_err


def _patch_app(app):
    """Patch the keeper's bridge accessor to return the fake (or None)."""
    return mock.patch.object(K, "_app", return_value=app)


class _ConfigGuard:
    """Set the two keeper flags on the live core.config and restore on exit."""
    def __init__(self, autostart=False, keep_open=False):
        self.autostart = autostart
        self.keep_open = keep_open

    def __enter__(self):
        import core.config as cfg
        self.cfg = cfg
        self._saved = (getattr(cfg, "APPLE_MUSIC_AUTOSTART", False),
                       getattr(cfg, "APPLE_MUSIC_KEEP_OPEN", False))
        cfg.APPLE_MUSIC_AUTOSTART = self.autostart
        cfg.APPLE_MUSIC_KEEP_OPEN = self.keep_open
        return cfg

    def __exit__(self, *exc):
        self.cfg.APPLE_MUSIC_AUTOSTART, self.cfg.APPLE_MUSIC_KEEP_OPEN = self._saved
        return False


# ─── staging gate ───────────────────────────────────────────────────────────
class StagingGateTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.get("JARVIS_STAGING")
        os.environ.pop("JARVIS_STAGING", None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        if self._saved_env is None:
            os.environ.pop("JARVIS_STAGING", None)
        else:
            os.environ["JARVIS_STAGING"] = self._saved_env

    def test_env_var_marks_staging(self):
        os.environ["JARVIS_STAGING"] = "1"
        self.assertTrue(K._is_staging())

    def test_monolith_flag_marks_staging(self):
        bc = types.ModuleType("bobert_companion")
        bc._is_staging = lambda: True
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            # ensure __main__ doesn't also define one that wins
            with mock.patch.object(K, "_bc", return_value=bc):
                self.assertTrue(K._is_staging())

    def test_not_staging_by_default(self):
        with mock.patch.object(K, "_bc", return_value=None):
            self.assertFalse(K._is_staging())


# ─── _launch_once ───────────────────────────────────────────────────────────
class LaunchOnceTests(unittest.TestCase):
    def setUp(self):
        self._stage = mock.patch.object(K, "_is_staging", return_value=False)
        self._stage.start()
        self.addCleanup(self._stage.stop)

    def test_launches_when_not_running(self):
        app = _FakeAppleMusic(running=False)
        with _patch_app(app):
            self.assertTrue(K._launch_once())
        self.assertEqual(app.launches, 1)

    def test_no_launch_when_already_running(self):
        app = _FakeAppleMusic(running=True)
        with _patch_app(app):
            self.assertFalse(K._launch_once())
        self.assertEqual(app.launches, 0)

    def test_no_launch_in_staging(self):
        app = _FakeAppleMusic(running=False)
        with mock.patch.object(K, "_is_staging", return_value=True), _patch_app(app):
            self.assertFalse(K._launch_once())
        self.assertEqual(app.launches, 0)

    def test_no_launch_when_bridge_absent(self):
        with _patch_app(None):
            self.assertFalse(K._launch_once())

    def test_is_running_raise_is_conservative(self):
        # If we can't tell whether it's running, do NOT blindly launch.
        app = _FakeAppleMusic(running_raises=True)
        with _patch_app(app):
            self.assertFalse(K._launch_once())
        self.assertEqual(app.launches, 0)

    def test_launch_failure_returns_false(self):
        app = _FakeAppleMusic(running=False, launch_ok=False, launch_err="boom")
        with _patch_app(app):
            self.assertFalse(K._launch_once())

    def test_launch_exception_swallowed(self):
        app = _FakeAppleMusic(running=False)
        app.launch = mock.MagicMock(side_effect=RuntimeError("explorer gone"))
        with _patch_app(app):
            self.assertFalse(K._launch_once())  # must not raise


# ─── _keep_alive_tick ───────────────────────────────────────────────────────
class KeepAliveTickTests(unittest.TestCase):
    def setUp(self):
        self._stage = mock.patch.object(K, "_is_staging", return_value=False)
        self._stage.start()
        self.addCleanup(self._stage.stop)

    def test_relaunches_only_when_dead(self):
        with _ConfigGuard(keep_open=True):
            app = _FakeAppleMusic(running=False)
            with _patch_app(app):
                self.assertTrue(K._keep_alive_tick())
            self.assertEqual(app.launches, 1)

    def test_noop_when_running(self):
        with _ConfigGuard(keep_open=True):
            app = _FakeAppleMusic(running=True)
            with _patch_app(app):
                self.assertFalse(K._keep_alive_tick())   # already up -> no launch
            self.assertEqual(app.launches, 0)

    def test_disabled_returns_none_and_never_launches(self):
        with _ConfigGuard(keep_open=False):
            app = _FakeAppleMusic(running=False)
            with _patch_app(app):
                self.assertIsNone(K._keep_alive_tick())
            self.assertEqual(app.launches, 0)

    def test_staging_returns_none(self):
        with _ConfigGuard(keep_open=True):
            app = _FakeAppleMusic(running=False)
            with mock.patch.object(K, "_is_staging", return_value=True), \
                 _patch_app(app):
                self.assertIsNone(K._keep_alive_tick())
            self.assertEqual(app.launches, 0)

    def test_bridge_absent_returns_none(self):
        with _ConfigGuard(keep_open=True):
            with _patch_app(None):
                self.assertIsNone(K._keep_alive_tick())


# ─── start_keeper (autostart + loop spawn + idempotency + gates) ────────────
class StartKeeperTests(unittest.TestCase):
    def setUp(self):
        self._stage = mock.patch.object(K, "_is_staging", return_value=False)
        self._stage.start()
        self.addCleanup(self._stage.stop)
        # Neuter Thread.start so no real daemon runs; record what was created.
        self._created = []
        real_thread_init = threading.Thread.__init__

        def rec_init(inst, *a, **k):
            real_thread_init(inst, *a, **k)
            self._created.append(inst)

        self._init_patch = mock.patch.object(threading.Thread, "__init__", rec_init)
        self._start_patch = mock.patch.object(threading.Thread, "start",
                                              lambda self: None)
        self._init_patch.start()
        self._start_patch.start()
        self.addCleanup(self._init_patch.stop)
        self.addCleanup(self._start_patch.stop)
        # Final safety net: even though Thread.start is neutered above, make sure
        # no real keep-alive watchdog escaped this test into the next one. Runs
        # AFTER the patches are torn down (addCleanup is LIFO), so stop_keeper's
        # join uses the real Thread machinery.
        self.addCleanup(self._assert_no_keeper_leak)

    def _assert_no_keeper_leak(self):
        K.stop_keeper(timeout=10.0)
        self.assertTrue(_no_keeper_thread_alive(),
                        "a real apple-music-keeper thread leaked out of the test")

    def _thread_names(self):
        return {t.name for t in self._created}

    def test_both_flags_off_is_noop(self):
        with _ConfigGuard(autostart=False, keep_open=False):
            self.assertFalse(K.start_keeper())
        self.assertEqual(self._created, [])

    def test_staging_is_hard_noop(self):
        with _ConfigGuard(autostart=True, keep_open=True), \
             mock.patch.object(K, "_is_staging", return_value=True):
            self.assertFalse(K.start_keeper())
        self.assertEqual(self._created, [])

    def test_autostart_only_spawns_launch_thread_no_loop(self):
        with _ConfigGuard(autostart=True, keep_open=False):
            self.assertFalse(K.start_keeper())   # no keep-alive loop started
        self.assertIn("apple-music-autostart", self._thread_names())
        self.assertNotIn(K._THREAD_NAME, self._thread_names())

    def test_keep_open_spawns_loop_thread(self):
        # Patch enumerate() to a clean list so the idempotency guard isn't
        # tripped by an unrelated live thread leaked by another test.
        with _ConfigGuard(autostart=False, keep_open=True), \
             mock.patch.object(threading, "enumerate", return_value=[]):
            self.assertTrue(K.start_keeper())
        self.assertIn(K._THREAD_NAME, self._thread_names())

    def test_both_spawn_autostart_and_loop(self):
        with _ConfigGuard(autostart=True, keep_open=True), \
             mock.patch.object(threading, "enumerate", return_value=[]):
            self.assertTrue(K.start_keeper())
        names = self._thread_names()
        self.assertIn("apple-music-autostart", names)
        self.assertIn(K._THREAD_NAME, names)

    def test_idempotent_does_not_double_spawn_loop(self):
        # Simulate the keep-alive thread already alive by name so the guard fires.
        fake_alive = mock.Mock()
        fake_alive.name = K._THREAD_NAME
        fake_alive.is_alive.return_value = True
        with _ConfigGuard(autostart=False, keep_open=True), \
             mock.patch.object(threading, "enumerate", return_value=[fake_alive]):
            self.assertFalse(K.start_keeper())   # skipped the duplicate
        self.assertNotIn(K._THREAD_NAME, self._thread_names())


# ─── stop_keeper (terminable loop + no leaked daemon) ───────────────────────
class StopKeeperTests(unittest.TestCase):
    """The keep-alive loop must be STOPPABLE — it waits on K._STOP instead of
    sleeping, so stop_keeper() makes it exit promptly and join. This is the real
    fix for the flaky leak: a stray watchdog can no longer outlive a test.

    These tests SPAWN A REAL loop thread on purpose, so the teardown is
    rigorous: it sets _STOP and then JOINS every spawned thread to death (no
    short timeout) before the test ends. That guarantee matters because this
    suite runs alongside ~12k other tests under heavy load — a thread merely
    "asked" to stop but not joined could otherwise outlive the test and flake a
    later one (e.g. the structural compile sweep)."""

    def setUp(self):
        self._spawned = []
        # LIFO cleanups: first reap our threads, THEN clear the event for the
        # next test. Reaping sets _STOP, so every loop we started exits.
        self.addCleanup(K._STOP.clear)
        self.addCleanup(self._reap_spawned)

    def _spawn(self, **mocks):
        """Start a REAL keep-alive loop thread (under the given K.* mocks if any
        are passed as already-started patchers) and TRACK it so teardown joins
        it to death. Returns the Thread."""
        t = threading.Thread(target=K._keep_alive_loop, daemon=True,
                             name=K._THREAD_NAME)
        self._spawned.append(t)
        t.start()
        return t

    def _reap_spawned(self):
        """Stop the loop and JOIN every thread we started until it's truly dead.
        No short timeout: once _STOP is set the loop returns after a single
        wait() wake (sub-millisecond of work), so even a CPU-starved thread
        finishes quickly once scheduled — we just must not give up early."""
        K._STOP.set()
        for t in self._spawned:
            # Generous, bounded join (not infinite, so a genuine hang still
            # surfaces) — far longer than the loop needs once _STOP is set.
            t.join(10.0)
            if t.is_alive():  # pragma: no cover - would indicate a real defect
                raise AssertionError(
                    "keep-alive loop thread failed to terminate after stop")

    def test_stop_keeper_with_nothing_running_is_true_and_safe(self):
        # No loop running → stop_keeper just sets the event and reports clean.
        K._STOP.clear()
        self.assertTrue(K.stop_keeper(timeout=0.5))
        self.assertTrue(K._STOP.is_set())

    def test_real_loop_thread_terminates_on_stop(self):
        # Spawn the REAL loop with tiny waits so the test is fast, then prove
        # stop_keeper() actually stops + joins it (no leaked daemon).
        K._STOP.clear()
        with mock.patch.object(K, "INITIAL_DELAY_SECONDS", 0.01), \
             mock.patch.object(K, "KEEP_ALIVE_INTERVAL_SECONDS", 0.02), \
             mock.patch.object(K, "_keep_alive_tick", return_value=None):
            t = self._spawn()
            # Let it spin through a few ticks so we're really mid-loop.
            time.sleep(0.1)
            self.assertTrue(t.is_alive())
            self.assertTrue(K.stop_keeper(timeout=10.0))
            self.assertFalse(t.is_alive())
        self.assertFalse(any(th.name == K._THREAD_NAME and th.is_alive()
                             for th in threading.enumerate()))

    def test_loop_exits_immediately_if_stopped_during_initial_delay(self):
        # If _STOP is set before/at the initial-delay wait, the loop returns
        # without ever ticking — no relaunch storm on a stop-during-startup.
        K._STOP.set()
        ticks = []
        with mock.patch.object(K, "INITIAL_DELAY_SECONDS", 5.0), \
             mock.patch.object(K, "_keep_alive_tick",
                               side_effect=lambda: ticks.append(1)):
            t = self._spawn()
            t.join(10.0)
        self.assertFalse(t.is_alive())     # exited promptly despite 5s delay
        self.assertEqual(ticks, [])        # never ticked

    def test_start_keeper_clears_stop_so_loop_runs(self):
        # A prior stop_keeper() set _STOP; start_keeper() must clear it so the
        # next loop actually runs instead of exiting on its first wait.
        K._STOP.set()
        with mock.patch.object(threading.Thread, "start", lambda self: None), \
             mock.patch.object(threading, "enumerate", return_value=[]), \
             _ConfigGuard(autostart=False, keep_open=True), \
             mock.patch.object(K, "_is_staging", return_value=False):
            self.assertTrue(K.start_keeper())
        self.assertFalse(K._STOP.is_set())   # cleared, ready for the real loop


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
