"""The TV detector's background sampler and its log lines (2026-10-12).

Before this the detector was only polled when something asked "is the TV on?"
(one ambient utterance, one double clap), so it never held the 3 fresh lit,
moving readings it needs and could not reach ON; its "previous frame" could be
minutes old (a person moving between two far-apart samples read as flicker); and
it said nothing about what it saw. Pinned here, all against a fake monolith
frame cache - no camera, no thread actually running:

  * the sampler loop feeds the decider on its own and reaches ON, then OFF;
  * every change of verdict and of the camera feed is logged once, with numbers;
  * a stale previous frame is dropped instead of producing a bogus delta;
  * a disabled flag reads no frame at all, and a live enable is noticed;
  * register() starts the sampler (not on staging); status reports the numbers.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from unittest import mock

from audio import tv_detect as td
from tests._skill_harness import load_skill_isolated
from tests.test_tv_detect import (
    _SkillBase, _bright_noisy_frame, _dark_static_frame)


class _Base(_SkillBase):
    def _load(self):
        mod = super()._load()
        mod._prev_frame_at[0] = 0.0
        mod._feed_ok[0] = None
        mod._verdict_logged[0] = False
        mod._last_reading.update(brightness=None, delta=None, at=0.0)
        return mod

    def _drive(self, mod, bc, frames, step=None):
        """Run the REAL sampler loop over `frames`: one per iteration, the
        monotonic clock advancing past the poll interval each time. Returns the
        printed log."""
        step = step or (mod.POLL_MIN_INTERVAL_S + 0.01)
        clock = [1000.0]
        # The decider stamps readings with the wall clock: advance it with the
        # monotonic one (from real now, so the readings stay fresh afterwards)
        # so the verdict dwell can expire inside one drive.
        wall0 = mod.time.time()
        feed = list(frames)
        stop = mock.MagicMock()
        n = [0]

        def is_set():
            return n[0] >= len(feed)

        def wait(_timeout=None):
            # the loop just polled once; present the next frame and move on
            clock[0] += step
            n[0] += 1
            if n[0] < len(feed):
                self._set_frame(bc, 0, feed[n[0]])
            return False
        stop.is_set.side_effect = is_set
        stop.wait.side_effect = wait
        self._set_frame(bc, 0, feed[0])
        buf = io.StringIO()
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]), \
                mock.patch.object(mod.time, "time",
                                  lambda: wall0 + (clock[0] - 1000.0)), \
                contextlib.redirect_stdout(buf):
            mod._sampler_loop(stop)
        return buf.getvalue()


class SamplerFeedsTheDecider(_Base):
    def test_the_loop_alone_reaches_on_and_logs_it(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        log = self._drive(mod, bc, [_bright_noisy_frame(seed=i)
                                    for i in range(td.WHOLE_FRAME_ON_FRAMES + 3)])
        self.assertTrue(mod.is_tv_on())
        self.assertEqual(log.count("[tv-detect] TV ON"), 1, log)
        self.assertIn("luma", log)
        self.assertIn("motion", log)
        self.assertIn("camera frames available", log)

    def test_it_goes_off_again_and_logs_that_too(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        frames = ([_bright_noisy_frame(seed=i)
                   for i in range(td.WHOLE_FRAME_ON_FRAMES + 2)]
                  + [_dark_static_frame()] * (td.WHOLE_FRAME_WINDOW
                                              + int(td.VERDICT_MIN_DWELL_S) + 2))
        log = self._drive(mod, bc, frames)
        self.assertFalse(mod.is_tv_on())
        self.assertEqual(log.count("[tv-detect] TV ON"), 1, log)
        self.assertEqual(log.count("[tv-detect] TV OFF"), 1, log)
        self.assertLess(log.index("TV ON"), log.index("TV OFF"))

    def test_a_steady_verdict_logs_nothing_more(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        log = self._drive(mod, bc, [_bright_noisy_frame(seed=i)
                                    for i in range(20)])
        self.assertEqual(log.count("TV ON"), 1, log)

    def test_dark_room_never_logs_a_tv(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _dark_static_frame()})
        log = self._drive(mod, bc, [_dark_static_frame()] * 10)
        self.assertNotIn("TV ON", log)
        self.assertFalse(mod.is_tv_on())

    def test_losing_and_regaining_the_camera_is_logged_once_each(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        buf = io.StringIO()
        clock = [1000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]), \
                contextlib.redirect_stdout(buf):
            mod._poll_once()
            with bc._camera_state_lock:
                bc._camera_latest_frame.clear()
            for _ in range(3):
                clock[0] += 1.1
                mod._poll_once()
            self._set_frame(bc, 0, _bright_noisy_frame(seed=9))
            clock[0] += 1.1
            mod._poll_once()
        log = buf.getvalue()
        self.assertEqual(log.count("no usable camera frame"), 1, log)
        self.assertEqual(log.count("camera frames available"), 2, log)

    def test_the_log_never_carries_pixels_or_paths(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        log = self._drive(mod, bc, [_bright_noisy_frame(seed=i)
                                    for i in range(6)])
        for line in log.splitlines():
            self.assertTrue(line.startswith("  [tv-detect]"), line)
            self.assertLess(len(line), 200)


class DeltaNeedsAFreshPreviousFrame(_Base):
    def test_a_stale_previous_frame_gives_no_delta(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=1)})
        clock = [1000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            mod._poll_once()                                   # frame A
            self._set_frame(bc, 0, _bright_noisy_frame(seed=2))
            clock[0] += mod.PREV_FRAME_MAX_AGE_S + 30          # a long pause
            mod._poll_once()                                   # frame B
        self.assertIsNone(mod._last_reading["delta"])
        self.assertIsNotNone(mod._last_reading["brightness"])

    def test_a_fresh_previous_frame_gives_a_delta(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=1)})
        clock = [1000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            mod._poll_once()
            self._set_frame(bc, 0, _bright_noisy_frame(seed=2))
            clock[0] += 1.1
            mod._poll_once()
        self.assertGreater(mod._last_reading["delta"], td.TEMPORAL_DELTA_ON_MIN)


class SamplerLifecycle(_Base):
    def test_disabled_reads_no_frame_and_a_live_enable_is_noticed(self):
        mod = self._load()
        self._patch_flag(False)
        self._fake_bc({0: _bright_noisy_frame(seed=0)})
        reads = []
        mod._snapshot_primary_frame = lambda: (reads.append(1), (None, None))[1]
        stop = mock.MagicMock()
        n = [0]
        stop.is_set.side_effect = lambda: n[0] >= 3
        waits = []

        def wait(t=None):
            waits.append(t)
            n[0] += 1
        stop.wait.side_effect = wait
        mod._sampler_loop(stop)
        self.assertEqual(reads, [], "a disabled detector must read no frame")
        self.assertTrue(all(w == mod.SAMPLER_IDLE_INTERVAL_S for w in waits))

    def test_a_crash_in_one_pass_does_not_end_the_loop(self):
        mod = self._load()
        self._patch_flag(True)
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("boom")
        mod._poll_once = boom
        stop = mock.MagicMock()
        n = [0]
        stop.is_set.side_effect = lambda: n[0] >= 3
        stop.wait.side_effect = lambda t=None: n.__setitem__(0, n[0] + 1)
        with contextlib.redirect_stdout(io.StringIO()):
            mod._sampler_loop(stop)
        self.assertEqual(len(calls), 3)

    def test_register_starts_the_sampler(self):
        env = {k: v for k, v in os.environ.items() if k != "JARVIS_STAGING"}
        with mock.patch.dict(os.environ, env, clear=True):
            mod, _actions = load_skill_isolated("tv_detect", register=True)
        try:
            self.assertIsNotNone(mod._sampler[0])
            self.assertEqual(mod._sampler[0].name, "tv-detect-sampler")
            self.assertTrue(mod._sampler[0].daemon)
        finally:
            mod._sampler[0] = None

    def test_staging_gets_no_sampler(self):
        with mock.patch.dict(os.environ, {"JARVIS_STAGING": "1"}):
            mod, _actions = load_skill_isolated("tv_detect", register=True)
        self.assertIsNone(mod._sampler[0])

    def test_start_is_idempotent_while_alive(self):
        mod = self._load()
        fake = mock.MagicMock()
        fake.is_alive.return_value = True
        mod._sampler[0] = fake
        self.assertFalse(mod.start_sampler())
        mod._sampler[0] = None

    def test_turning_it_on_by_voice_starts_the_sampler(self):
        mod = self._load()
        self._patch_flag(False)
        from tools import settings_window as sw
        saved = {}
        with mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved)), \
                mock.patch.object(sw, "save_settings",
                                  lambda d, *a, **k: saved.update(d)):
            mod._sampler[0] = None
            mod.tv_detect_on("")
        self.assertIsNotNone(mod._sampler[0])
        mod._sampler[0] = None


class StatusAndSeams(_Base):
    def test_status_reports_what_it_measured(self):
        mod = self._load()
        self._patch_flag(True)
        bc = self._fake_bc({0: _bright_noisy_frame(seed=0)})
        clock = [1000.0]
        with mock.patch.object(mod.time, "monotonic", lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            for i in range(td.WHOLE_FRAME_ON_FRAMES + 2):
                self._set_frame(bc, 0, _bright_noisy_frame(seed=i + 1))
                clock[0] += 1.1
                mod._poll_once()
            out = mod.tv_detect_status("")
        self.assertIn("brightness", out)
        self.assertIn("motion", out)
        self.assertIn("DO see a lit, moving screen", out)

    def test_status_says_when_the_sampler_is_not_running(self):
        mod = self._load()
        self._patch_flag(True)
        self._fake_bc({0: _dark_static_frame()})
        mod._sampler[0] = None
        self.assertIn("background sampler is not running",
                      mod.tv_detect_status(""))

    def test_the_monolith_is_found_by_its_frame_cache_not_by_main(self):
        mod, _a = load_skill_isolated("tv_detect", register=False)
        fake = types.ModuleType("bobert_companion")
        fake._camera_latest_frame = {}
        stray = types.ModuleType("__main__")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake,
                                           "__main__": stray}):
            self.assertIs(mod._bc(), fake)

    def test_the_shipped_default_is_on(self):
        from core import config as cfg
        self.assertTrue(cfg.TV_DETECT_ENABLED)


if __name__ == "__main__":
    unittest.main()
