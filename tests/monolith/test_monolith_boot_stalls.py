"""reliability-11 (2026-11-21) - boot stalls on the path to 'ready to listen'.

* BOOT_AUDIO_PROC_PREWARM: the audio processor (noisereduce -> torch + scipy,
  7.4 s on the 11-20 boot thread) builds on a daemon thread while the skills
  load and is JOINED right after load_skills(), before the camera probe and
  the face tracker, so scipy's DLL load still never overlaps a camera open.
  Off = built on the boot thread just before the face tracker, as before.
* load_skills prints '[skill-load] slow: <name> <secs> s' and one summary
  line (log only).

Everything external is mocked: no audio processor build, no camera."""
from __future__ import annotations

import inspect
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


class _FakeAP:
    """Stands in for core.audio_processor: get_processor() runs `body`."""

    def __init__(self, body):
        self.calls = 0
        self._body = body

    def get_processor(self, *a, **k):
        self.calls += 1
        return self._body(self.calls)


@requires_monolith
class AudioProcPrewarmTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_is_staging", return_value=False)
        self._p(bc, "BOOT_AUDIO_PROC_PREWARM", True, create=True)
        p = mock.patch.dict(bc._audio_proc_prewarm,
                            {"thread": None, "ok": False, "ms": 0.0,
                             "err": "", "waited_ms": 0.0})
        p.start()
        self.addCleanup(p.stop)

    def _p(self, *a, **k):
        p = mock.patch.object(*a, **k)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def _install(self, fake):
        import core
        self._p(core, "audio_processor", fake, create=True)
        p = mock.patch.dict("sys.modules", {"core.audio_processor": fake})
        p.start()
        self.addCleanup(p.stop)

    def test_flags_default_on_and_logged(self):
        import core.config as cfg
        self.assertIs(cfg.BOOT_AUDIO_PROC_PREWARM, True)
        self.assertIs(cfg.AMAZON_TRACKER_DEFER_BACKEND_CHECK, True)
        self.assertIn("BOOT_AUDIO_PROC_PREWARM", self.bc._TURN_FLAG_KEYS)
        self.assertIn("AMAZON_TRACKER_DEFER_BACKEND_CHECK",
                      self.bc._TURN_FLAG_KEYS)
        self.assertTrue(self.bc._audio_proc_prewarm_on())
        for v, want in (("off", False), ("on", True), (False, False)):
            self.bc.BOOT_AUDIO_PROC_PREWARM = v
            self.assertIs(self.bc._audio_proc_prewarm_on(), want)

    def test_start_returns_before_the_build_and_join_waits_for_it(self):
        bc = self.bc
        release = threading.Event()
        built = threading.Event()

        def body(n):
            release.wait(10)
            built.set()
            return object()

        fake = _FakeAP(body)
        self._install(fake)
        t0 = time.monotonic()
        self.assertTrue(bc._audio_proc_prewarm_start())
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertFalse(built.is_set())          # boot thread is free
        threading.Timer(0.2, release.set).start()
        bc._audio_proc_prewarm_join()
        self.assertTrue(built.is_set())            # joined: build finished
        self.assertTrue(bc._audio_proc_prewarm["ok"])
        self.assertGreaterEqual(bc._audio_proc_prewarm["waited_ms"], 100)
        with mock.patch("builtins.print") as pr:
            bc._audio_proc_boot_ready()
        self.assertEqual(fake.calls, 1)            # not built a second time
        printed = " ".join(str(c) for c in pr.call_args_list)
        self.assertIn("built on a thread", printed)

    def test_flag_off_builds_on_the_boot_thread_as_before(self):
        bc = self.bc
        bc.BOOT_AUDIO_PROC_PREWARM = False
        fake = _FakeAP(lambda n: object())
        self._install(fake)
        self.assertFalse(bc._audio_proc_prewarm_start())
        bc._audio_proc_prewarm_join()              # no thread: no-op
        self.assertEqual(fake.calls, 0)
        with mock.patch("builtins.print") as pr:
            bc._audio_proc_boot_ready()
        self.assertEqual(fake.calls, 1)
        printed = " ".join(str(c) for c in pr.call_args_list)
        self.assertIn("audio processor ready before the camera opens", printed)
        self.assertNotIn("built on a thread", printed)

    def test_staging_never_starts_the_thread(self):
        self.bc._is_staging.return_value = True
        self._install(_FakeAP(lambda n: object()))
        self.assertFalse(self.bc._audio_proc_prewarm_start())

    def test_failed_thread_falls_back_to_the_synchronous_build(self):
        bc = self.bc

        def body(n):
            if n == 1:
                raise OSError("dll load failed")
            return object()

        fake = _FakeAP(body)
        self._install(fake)
        bc._audio_proc_prewarm_start()
        bc._audio_proc_prewarm_join()
        self.assertFalse(bc._audio_proc_prewarm["ok"])
        with mock.patch("builtins.print") as pr:
            bc._audio_proc_boot_ready()
        self.assertEqual(fake.calls, 2)
        printed = " ".join(str(c) for c in pr.call_args_list)
        self.assertIn("background build failed", printed)
        self.assertIn("ready before the camera opens", printed)

    def test_import_lock_race_is_retried_once(self):
        bc = self.bc

        class _DeadlockError(RuntimeError):
            pass

        def body(n):
            if n == 1:
                raise _DeadlockError("deadlock detected by _ModuleLock")
            return object()

        fake = _FakeAP(body)
        self._install(fake)
        with mock.patch.object(bc.time, "sleep"):
            bc._audio_proc_prewarm_work()
        self.assertEqual(fake.calls, 2)
        self.assertTrue(bc._audio_proc_prewarm["ok"])

    def test_main_joins_before_any_camera_can_open(self):
        src = inspect.getsource(self.bc.main)
        order = ["_audio_proc_prewarm_start()", "load_skills()",
                 "_audio_proc_prewarm_join()",
                 "probe_cameras_and_update_config()",
                 "_audio_proc_boot_ready()",
                 "target=_face_tracking_thread_after_boot"]
        pos = [src.find(s) for s in order]
        self.assertNotIn(-1, pos, dict(zip(order, pos)))
        self.assertEqual(pos, sorted(pos), dict(zip(order, pos)))
        # nothing runs between load_skills() and the join
        between = src[pos[1] + len(order[1]):pos[2]]
        self.assertEqual(between.strip(), "")


_FAST = "def register(actions):\n    actions['fast_probe'] = lambda a='': 'ok'\n"
_BROKEN = "raise RuntimeError('broken skill')\n"


@requires_monolith
class SkillLoadTimingTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for name, body in (("aa_fast", _FAST), ("bb_broken", _BROKEN)):
            with open(os.path.join(self._tmp.name, f"{name}.py"), "w",
                      encoding="utf-8") as f:
                f.write(body)
        for k, v in (("SKILLS_DIR", self._tmp.name), ("SKILLS_ENABLED", True),
                     ("_loaded_skill_names", set()),
                     ("ACTIONS", dict(bc.ACTIONS)),
                     ("_skill_load_times", [])):
            p = mock.patch.object(bc, k, v)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: [bc.sys.modules.pop(f"skill_{n}", None)
                                 for n in ("aa_fast", "bb_broken")])

    def _load(self, slow_s):
        with mock.patch.object(self.bc, "SKILL_LOAD_SLOW_S", slow_s), \
                mock.patch("builtins.print") as pr:
            self.bc.load_skills()
        return [str(c.args[0]) if c.args else "" for c in pr.call_args_list]

    def test_slow_skills_are_named_and_a_summary_closes_the_pass(self):
        lines = self._load(0.0)
        slow = [ln for ln in lines if "[skill-load] slow:" in ln]
        self.assertEqual(len(slow), 2)             # the broken one is timed too
        self.assertTrue(any("aa_fast" in ln for ln in slow))
        self.assertTrue(any("bb_broken" in ln for ln in slow))
        summary = [ln for ln in lines if "[skill-load] 2 skill(s) in" in ln]
        self.assertEqual(len(summary), 1)
        self.assertIn("slowest:", summary[0])
        self.assertEqual(self.bc._skill_load_times, [])   # reset per pass
        self.assertIn("fast_probe", self.bc.ACTIONS)      # loading unchanged

    def test_fast_skills_print_only_the_summary(self):
        lines = self._load(1e9)
        self.assertFalse(any("[skill-load] slow:" in ln for ln in lines))
        self.assertEqual(
            sum("[skill-load] 2 skill(s) in" in ln for ln in lines), 1)


if __name__ == "__main__":
    unittest.main()
