"""Unit tests for SECTION 3 of core/actions.py — the ``_act_*`` action
handlers defined between lines 998 and 2086 of that module.

That band covers:

  * maintenance / lifecycle   _act_force_backup / _act_reset_memory /
                              _act_version_info / _act_run_smoke_test /
                              _act_test_each_skill / _act_forget_last_hour /
                              _act_latency_benchmark
  * music routing             _act_play_music
  * webcam awareness          _act_where_is_user / _act_see_user /
                              _act_which_monitor
  * vision capture + recall   _act_see_screen / _act_recall_screen
  * replay                    _act_replay_last_action
  * shell                     _act_run_shell
  * session recall            _act_session_memory_recall
  * changelog                 _act_read_changelog
  * overnight + upgrade        _act_start_overnight_upgrade / _act_upgrade
  * window placement          _act_open_on_monitor / _act_move_window_to_monitor
  * skill authoring           _act_create_skill

EVERYTHING external is mocked so the suite is CI-faithful (runs, never
skips, under ``tools/run_tests_ci_sim.py`` where heavy deps are blocked and
the host is made to look like Linux):

  * The ~14K-line ``bobert_companion`` monolith is NEVER imported — each
    handler reaches it through ``core.actions._bc()``, which we patch with
    ``mock.patch.object(A, "_bc", return_value=fake)`` to hand back a
    configured ``Mock``. No ``@requires_monolith``.
  * Heavy per-handler imports absent on CI (``cv2``, ``pygetwindow``,
    ``win32gui``/``win32con``) are injected as fakes into ``sys.modules``
    via ``mock.patch.dict`` so the handler body runs faithfully and the
    fake auto-restores after the test.
  * ``subprocess.run`` / ``subprocess.Popen`` / ``os.startfile`` /
    ``threading.Thread`` / ``time.sleep`` are all mocked; real hardware,
    network, processes and threads are never touched.
  * Filesystem writes go to ``tempfile`` dirs; ``time`` is frozen where the
    output depends on the clock.

Stdlib ``unittest`` + ``unittest.mock`` only (no pytest). Per-test patches
auto-restore; module-level ``core.actions`` globals are not mutated. Real
numpy is left intact (it is never imported here).
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import core.actions as A


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _patch_bc(fake):
    """Context manager: make ``A._bc()`` return ``fake`` for the duration."""
    return mock.patch.object(A, "_bc", return_value=fake)


def _base_bc(tmpdir=None):
    """A fresh Mock standing in for the bobert_companion monolith.

    ``__file__`` is pointed at a real directory so handlers that derive
    sibling paths from ``os.path.dirname(bc.__file__)`` resolve inside a
    tempdir rather than the live project root.
    """
    bc = mock.Mock()
    root = tmpdir or tempfile.gettempdir()
    bc.__file__ = os.path.join(root, "bobert_companion.py")
    return bc


@contextlib.contextmanager
def _only_skill_modules(fakes):
    """Make ``sys.modules`` expose EXACTLY the given ``skill_*`` entries while
    the block runs, hiding any real ``skill_*`` modules a sibling test module
    may have registered. Non-skill modules are untouched. Everything is
    restored on exit.

    ``_act_test_each_skill`` enumerates ``sys.modules`` for names starting
    with ``skill_``; without this isolation the full-suite / ci-sim run would
    leak real skill modules into the sweep and break exact-count assertions.
    """
    removed = {k: sys.modules[k] for k in list(sys.modules)
               if k.startswith("skill_")}
    for k in removed:
        del sys.modules[k]
    sys.modules.update(fakes)
    try:
        yield
    finally:
        for k in list(fakes):
            sys.modules.pop(k, None)
        sys.modules.update(removed)


class _CaptureTrayAsync:
    """Stand-in for ``bc._tray_async(label, fn)`` that records the label and
    the submitted callable WITHOUT running it (the real one hands the work to
    a tray executor thread). Tests then invoke ``.fn()`` directly to exercise
    the closure body deterministically, in-process."""

    def __init__(self):
        self.label = None
        self.fn = None
        self.calls = 0

    def __call__(self, label, fn):
        self.calls += 1
        self.label = label
        self.fn = fn
        return None


# ===========================================================================
# _act_force_backup
# ===========================================================================
class ForceBackupTests(unittest.TestCase):
    def test_returns_immediately_and_submits_async(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        with _patch_bc(bc):
            out = A._act_force_backup()
        self.assertEqual(out, "backup started")
        self.assertEqual(tray.label, "force_backup")
        self.assertIsNotNone(tray.fn)

    def test_inner_do_missing_upgrade_script(self):
        # __file__ points at a tempdir with no upgrade_jarvis.py beside it.
        with tempfile.TemporaryDirectory() as td:
            bc = _base_bc(td)
            tray = _CaptureTrayAsync()
            bc._tray_async = tray
            with _patch_bc(bc):
                A._act_force_backup()
            self.assertEqual(tray.fn(), "upgrade_jarvis.py not found")

    def test_inner_do_runs_backup_and_reports_basename(self):
        with tempfile.TemporaryDirectory() as td:
            # Write a stub upgrade_jarvis.py whose backup_codebase() returns a path.
            dest = os.path.join(td, "backups", "snap_20260101.zip")
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write(
                    "def backup_codebase():\n"
                    f"    return {dest!r}\n"
                )
            bc = _base_bc(td)
            tray = _CaptureTrayAsync()
            bc._tray_async = tray
            with _patch_bc(bc):
                A._act_force_backup()
            self.assertEqual(tray.fn(), "backup -> snap_20260101.zip")

    def test_inner_do_reports_exception(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write("def backup_codebase():\n    raise RuntimeError('disk full')\n")
            bc = _base_bc(td)
            tray = _CaptureTrayAsync()
            bc._tray_async = tray
            with _patch_bc(bc):
                A._act_force_backup()
            self.assertEqual(tray.fn(), "backup failed: disk full")


# ===========================================================================
# _act_reset_memory
# ===========================================================================
class ResetMemoryTests(unittest.TestCase):
    def _bc_with_memory(self, td, write_file=True):
        bc = _base_bc(td)
        bc._memory_lock = mock.MagicMock()  # supports `with`
        mem_path = os.path.join(td, "bobert_memory.json")
        if write_file:
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write('{"facts": ["x"]}')
        bc.MEMORY_FILE = mem_path
        bc._empty_memory.return_value = {"facts": []}
        return bc, mem_path

    def test_backs_up_then_resets_when_file_exists(self):
        with tempfile.TemporaryDirectory() as td:
            bc, mem_path = self._bc_with_memory(td)
            with _patch_bc(bc):
                out = A._act_reset_memory()
            self.assertIn("memory reset (backup -> backups/memory_pre_reset_",
                          out)
            bc.save_memory.assert_called_once_with({"facts": []})
            # A backup copy was actually written into backups/.
            backups = os.listdir(os.path.join(td, "backups"))
            self.assertTrue(any(b.startswith("memory_pre_reset_")
                                for b in backups))

    def test_no_file_reports_already_empty(self):
        with tempfile.TemporaryDirectory() as td:
            bc, _ = self._bc_with_memory(td, write_file=False)
            with _patch_bc(bc):
                out = A._act_reset_memory()
            self.assertEqual(out, "memory was already empty")
            bc.save_memory.assert_called_once()

    def test_copy_failure_refuses_to_wipe(self):
        with tempfile.TemporaryDirectory() as td:
            bc, _ = self._bc_with_memory(td)
            with _patch_bc(bc), \
                    mock.patch.object(A.shutil, "copy2",
                                      side_effect=OSError("perm denied")):
                out = A._act_reset_memory()
            self.assertTrue(out.startswith("backup failed, refused to wipe:"))
            bc.save_memory.assert_not_called()

    def test_outer_exception_caught(self):
        bc = _base_bc()
        # _memory_lock that explodes on __enter__ triggers the outer except.
        lock = mock.MagicMock()
        lock.__enter__.side_effect = RuntimeError("lock boom")
        bc._memory_lock = lock
        bc.MEMORY_FILE = "irrelevant.json"
        with _patch_bc(bc):
            out = A._act_reset_memory()
        self.assertTrue(out.startswith("reset_memory failed: lock boom"))


# ===========================================================================
# _act_version_info
# ===========================================================================
class VersionInfoTests(unittest.TestCase):
    def _write_version(self, td, payload):
        data_dir = os.path.join(td, "data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "version.json"), "w",
                  encoding="utf-8") as f:
            import json
            json.dump(payload, f)

    def test_no_version_file_uses_release_version(self):
        with tempfile.TemporaryDirectory() as td:
            bc = _base_bc(td)
            with _patch_bc(bc):
                out = A._act_version_info()
            # release_ver comes from core.version (real, importable on CI).
            self.assertIn("I'm on version", out)
            self.assertTrue(out.endswith("sir."))

    def test_no_timestamp_field(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_version(td, {})
            bc = _base_bc(td)
            with _patch_bc(bc):
                out = A._act_version_info()
            self.assertIn("no upgrade timestamp on file", out)

    def test_unparseable_timestamp(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_version(td, {"last_upgrade_at": "not-a-date"})
            bc = _base_bc(td)
            with _patch_bc(bc):
                out = A._act_version_info()
            self.assertIn("last updated not-a-date", out)

    def test_same_day_morning_phrasing(self):
        # Freeze "now" and use a timestamp earlier the same day at 08:43.
        from datetime import datetime
        fixed_now = datetime(2026, 6, 1, 15, 0, 0)
        ts = datetime(2026, 6, 1, 8, 43, 0)
        with tempfile.TemporaryDirectory() as td:
            self._write_version(td, {"last_upgrade_at": ts.isoformat()})
            bc = _base_bc(td)

            class _FrozenDT(datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed_now

            with _patch_bc(bc), \
                    mock.patch("datetime.datetime", _FrozenDT):
                out = A._act_version_info()
            self.assertIn("this morning at 8:43 AM", out)

    def test_yesterday_phrasing(self):
        from datetime import datetime
        fixed_now = datetime(2026, 6, 2, 10, 0, 0)
        ts = datetime(2026, 6, 1, 20, 15, 0)  # 8:15 PM -> "evening"
        with tempfile.TemporaryDirectory() as td:
            self._write_version(td, {"last_upgrade_at": ts.isoformat()})
            bc = _base_bc(td)

            class _FrozenDT(datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed_now

            with _patch_bc(bc), \
                    mock.patch("datetime.datetime", _FrozenDT):
                out = A._act_version_info()
            self.assertIn("yesterday evening at 8:15 PM", out)

    def test_within_week_uses_weekday(self):
        from datetime import datetime
        fixed_now = datetime(2026, 6, 8, 10, 0, 0)   # Monday
        ts = datetime(2026, 6, 5, 13, 0, 0)          # Friday afternoon, 3 days
        with tempfile.TemporaryDirectory() as td:
            self._write_version(td, {"last_upgrade_at": ts.isoformat()})
            bc = _base_bc(td)

            class _FrozenDT(datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed_now

            with _patch_bc(bc), \
                    mock.patch("datetime.datetime", _FrozenDT):
                out = A._act_version_info()
            self.assertIn("Friday afternoon at 1:00 PM", out)

    def test_older_than_week_uses_month_day(self):
        from datetime import datetime
        fixed_now = datetime(2026, 6, 30, 10, 0, 0)
        ts = datetime(2026, 6, 1, 23, 30, 0)  # night
        with tempfile.TemporaryDirectory() as td:
            self._write_version(td, {"last_upgrade_at": ts.isoformat()})
            bc = _base_bc(td)

            class _FrozenDT(datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed_now

            with _patch_bc(bc), \
                    mock.patch("datetime.datetime", _FrozenDT):
                out = A._act_version_info()
            self.assertIn("on June 01 at 11:30 PM", out)

    def test_outer_exception_caught(self):
        # bc.__file__ as a non-string makes os.path.dirname raise -> outer except.
        bc = mock.Mock()
        bc.__file__ = 1234
        with _patch_bc(bc):
            out = A._act_version_info()
        self.assertTrue(out.startswith("could not read version info:"))

    def test_core_version_import_failure_uses_unknown(self):
        # Force `from core.version import __version__` to raise -> "unknown".
        with tempfile.TemporaryDirectory() as td:
            bc = _base_bc(td)  # no data/version.json -> early return branch
            real_import = __import__

            def _imp(name, *a, **k):
                if name == "core.version":
                    raise ImportError("boom")
                return real_import(name, *a, **k)

            with _patch_bc(bc), \
                    mock.patch("builtins.__import__", side_effect=_imp):
                out = A._act_version_info()
            self.assertIn("I'm on version unknown, sir.", out)


# ===========================================================================
# _act_run_smoke_test
# ===========================================================================
class RunSmokeTestTests(unittest.TestCase):
    def test_returns_running_and_submits_async(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        with _patch_bc(bc):
            out = A._act_run_smoke_test()
        self.assertEqual(out, "smoke test running")
        self.assertEqual(tray.label, "run_smoke_test")

    def test_inner_do_all_clean(self):
        with tempfile.TemporaryDirectory() as td:
            # Provide one compilable target file (bobert_companion.py).
            with open(os.path.join(td, "bobert_companion.py"), "w",
                      encoding="utf-8") as f:
                f.write("x = 1\n")
            bc = _base_bc(td)
            tray = _CaptureTrayAsync()
            bc._tray_async = tray
            with _patch_bc(bc):
                A._act_run_smoke_test()
            out = tray.fn()
            self.assertTrue(out.startswith("smoke test PASSED"))
            self.assertIn("1 files clean", out)

    def test_inner_do_reports_syntax_failure(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "bobert_companion.py"), "w",
                      encoding="utf-8") as f:
                f.write("def broken(:\n")  # syntax error
            bc = _base_bc(td)
            tray = _CaptureTrayAsync()
            bc._tray_async = tray
            with _patch_bc(bc):
                A._act_run_smoke_test()
            out = tray.fn()
            self.assertTrue(out.startswith("smoke test FAILED"))

    def test_inner_do_oserror_during_compile(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "bobert_companion.py"), "w",
                      encoding="utf-8") as f:
                f.write("ok = 1\n")
            bc = _base_bc(td)
            tray = _CaptureTrayAsync()
            bc._tray_async = tray
            with _patch_bc(bc):
                A._act_run_smoke_test()
            # py_compile raises OSError (e.g. unreadable file) -> reported as repr.
            import py_compile
            with mock.patch.object(py_compile, "compile",
                                   side_effect=OSError("io fail")):
                out = tray.fn()
            self.assertTrue(out.startswith("smoke test FAILED"))
            self.assertIn("OSError", out)

    def test_inner_do_outer_exception(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        with _patch_bc(bc):
            A._act_run_smoke_test()
        # Make os.path.dirname raise inside _do -> outer "errored" branch.
        with mock.patch.object(A.os.path, "dirname",
                               side_effect=RuntimeError("dirfail")):
            out = tray.fn()
        self.assertTrue(out.startswith("smoke test errored: dirfail"))

    def test_inner_do_includes_skill_files(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "bobert_companion.py"), "w",
                      encoding="utf-8") as f:
                f.write("ok = True\n")
            skills = os.path.join(td, "skills")
            os.makedirs(skills)
            with open(os.path.join(skills, "good.py"), "w",
                      encoding="utf-8") as f:
                f.write("y = 2\n")
            # _-prefixed file is skipped by the loader.
            with open(os.path.join(skills, "_private.py"), "w",
                      encoding="utf-8") as f:
                f.write("nope(\n")
            bc = _base_bc(td)
            tray = _CaptureTrayAsync()
            bc._tray_async = tray
            with _patch_bc(bc):
                A._act_run_smoke_test()
            out = tray.fn()
            self.assertTrue(out.startswith("smoke test PASSED"))
            self.assertIn("2 files clean", out)  # bobert_companion + good.py


# ===========================================================================
# _act_test_each_skill
# ===========================================================================
class TestEachSkillTests(unittest.TestCase):
    def test_announce_counts_loaded_skill_modules(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        fake_mods = {"skill_alpha": mock.Mock(), "skill_beta": mock.Mock()}
        with _patch_bc(bc), _only_skill_modules(fake_mods):
            out = A._act_test_each_skill()
        self.assertEqual(out, "testing 2 skill module(s)")
        self.assertEqual(tray.label, "test_each_skill")

    def test_inner_do_no_skills(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        # Ensure NO skill_* modules are present for the inner sweep.
        with _patch_bc(bc), _only_skill_modules({}):
            A._act_test_each_skill()
            out = tray.fn()
        self.assertEqual(out, "no skills loaded")

    def test_inner_do_ok_fail_silent_mix(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray

        ok_mod = types.ModuleType("skill_ok")
        ok_mod.selftest = lambda: True
        fail_bool = types.ModuleType("skill_failbool")
        fail_bool.selftest = lambda: False
        fail_dict = types.ModuleType("skill_faildict")
        fail_dict.selftest = lambda: {"ok": False, "why": "nope"}
        raiser = types.ModuleType("skill_raiser")
        def _boom():
            raise ValueError("kaboom")
        raiser.selftest = _boom
        silent = types.ModuleType("skill_silent")  # no selftest attr

        mods = {
            "skill_ok": ok_mod, "skill_failbool": fail_bool,
            "skill_faildict": fail_dict, "skill_raiser": raiser,
            "skill_silent": silent,
        }
        with _patch_bc(bc), _only_skill_modules(mods):
            A._act_test_each_skill()
            out = tray.fn()
        self.assertIn("1 OK", out)
        self.assertIn("3 FAIL", out)
        self.assertIn("1 no selftest", out)
        self.assertIn("failed:", out)

    def test_inner_do_skips_none_module_entry(self):
        # A skill_* key present in sys.modules but mapped to None is skipped.
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        ok_mod = types.ModuleType("skill_real")
        ok_mod.selftest = lambda: True
        mods = {"skill_real": ok_mod, "skill_phantom": None}
        with _patch_bc(bc), _only_skill_modules(mods):
            A._act_test_each_skill()
            out = tray.fn()
        # Only the real module counted; the None entry neither OK nor silent.
        self.assertIn("1 OK", out)
        self.assertIn("0 FAIL", out)
        self.assertIn("0 no selftest", out)


# ===========================================================================
# _act_forget_last_hour
# ===========================================================================
class ForgetLastHourTests(unittest.TestCase):
    def _bc(self, mem):
        bc = _base_bc()
        bc._memory_lock = mock.MagicMock()
        bc.load_memory.return_value = mem
        return bc

    def test_drops_recent_topics_and_sessions(self):
        # Freeze time so the cutoff is deterministic. now = 2026-06-01 12:00.
        fixed = 1_000_000.0
        mem = {
            "topics": [
                {"date": "2026-06-01 11:50"},   # within last hour -> dropped
                {"date": "2026-01-01 09:00"},   # old -> kept
            ],
            "sessions": [
                {"date": "2026-06-01 11:30"},   # dropped
            ],
            "facts": ["durable"],
        }
        bc = self._bc(mem)
        # cutoff string is strftime of now-3600; pin both.
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=fixed), \
                mock.patch.object(A.time, "localtime",
                                  return_value=__import__("time").struct_time(
                                      (2026, 6, 1, 11, 0, 0, 0, 152, -1))), \
                mock.patch.object(A.time, "strftime",
                                  return_value="2026-06-01 11:00"):
            out = A._act_forget_last_hour()
        self.assertEqual(out, "forgot 2 item(s) from the last hour")
        saved = bc.save_memory.call_args[0][0]
        self.assertEqual(len(saved["topics"]), 1)
        self.assertEqual(saved["sessions"], [])
        # Facts untouched.
        self.assertEqual(saved["facts"], ["durable"])

    def test_nothing_recent(self):
        mem = {"topics": [{"date": "2020-01-01 00:00"}], "sessions": []}
        bc = self._bc(mem)
        with _patch_bc(bc), \
                mock.patch.object(A.time, "strftime",
                                  return_value="2026-06-01 11:00"), \
                mock.patch.object(A.time, "localtime"), \
                mock.patch.object(A.time, "time", return_value=0.0):
            out = A._act_forget_last_hour()
        self.assertEqual(out, "nothing recent enough to forget")
        bc.save_memory.assert_not_called()

    def test_exception_caught(self):
        bc = _base_bc()
        bc._memory_lock = mock.MagicMock()
        bc.load_memory.side_effect = RuntimeError("mem broke")
        with _patch_bc(bc), \
                mock.patch.object(A.time, "strftime", return_value="x"), \
                mock.patch.object(A.time, "localtime"), \
                mock.patch.object(A.time, "time", return_value=0.0):
            out = A._act_forget_last_hour()
        self.assertTrue(out.startswith("forget_last_hour failed: mem broke"))


# ===========================================================================
# _act_latency_benchmark
# ===========================================================================
class LatencyBenchmarkTests(unittest.TestCase):
    def test_returns_running_and_submits_async(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        with _patch_bc(bc):
            out = A._act_latency_benchmark()
        self.assertEqual(out, "latency benchmark running")
        self.assertEqual(tray.label, "latency_benchmark")

    def test_inner_do_reports_latency(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        bc._llm_quick.return_value = "pong\nextra"
        # time.time first call = t0, second = t0 + 0.05s -> 50ms.
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", side_effect=[100.0, 100.05]):
            A._act_latency_benchmark()
            out = tray.fn()
        self.assertIn("50ms", out)
        self.assertIn("reply='pong'", out)

    def test_inner_do_handles_exception(self):
        bc = _base_bc()
        tray = _CaptureTrayAsync()
        bc._tray_async = tray
        bc._llm_quick.side_effect = RuntimeError("backend down")
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=0.0):
            A._act_latency_benchmark()
            out = tray.fn()
        self.assertTrue(out.startswith("latency_benchmark failed: backend down"))


# ===========================================================================
# _act_play_music
# ===========================================================================
class PlayMusicTests(unittest.TestCase):
    def test_library_prefix_forces_local(self):
        bc = _base_bc()
        bc._play_music_core.return_value = (True, "playing Earth Song")
        with _patch_bc(bc):
            out = A._act_play_music("library: Earth Song")
        self.assertEqual(out, "playing Earth Song")
        bc._play_music_core.assert_called_once_with("Earth Song", force=True)

    def test_reroutes_to_apple_music_when_chrome_active(self):
        bc = _base_bc()
        bc._apple_music_chrome_active.return_value = True
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_apple_music",
                                  return_value="streamed via apple music") as am:
            out = A._act_play_music("Smooth Criminal")
        self.assertEqual(out, "streamed via apple music")
        am.assert_called_once_with("Smooth Criminal")
        bc._play_music_core.assert_not_called()

    def test_default_path_uses_itunes_core(self):
        bc = _base_bc()
        bc._apple_music_chrome_active.return_value = False
        bc._play_music_core.return_value = (False, "no tracks found")
        with _patch_bc(bc):
            out = A._act_play_music("Thriller")
        self.assertEqual(out, "no tracks found")
        bc._play_music_core.assert_called_once_with("Thriller")


# ===========================================================================
# _act_where_is_user
# ===========================================================================
class WhereIsUserTests(unittest.TestCase):
    def _cams(self):
        return [
            {"index": 0, "label": "left cam", "look_x": 0.2},
            {"index": 1, "label": "right cam", "look_x": 0.8},
        ]

    def _bc(self, last_seen=None, errors=None, error_at=None):
        bc = _base_bc()
        bc._camera_state_lock = mock.MagicMock()
        bc._camera_last_seen = last_seen or {}
        bc._camera_last_read_error = errors or {}
        bc._camera_last_read_error_at = error_at or {}
        return bc

    def test_no_cameras_configured(self):
        bc = self._bc()
        with _patch_bc(bc), \
                mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch("core.config.CAMERAS", []):
            out = A._act_where_is_user()
        self.assertEqual(out, "no cameras configured")

    def test_visible_to_all_cameras(self):
        now = 10_000.0
        bc = self._bc(last_seen={0: now - 1.0, 1: now - 1.5})
        with _patch_bc(bc), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_where_is_user()
        self.assertIn("visible to ALL cameras", out)
        self.assertIn("sees user NOW", out)

    def test_visible_to_only_one(self):
        now = 10_000.0
        bc = self._bc(last_seen={0: now - 1.0, 1: now - 50.0})
        with _patch_bc(bc), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_where_is_user()
        self.assertIn("visible only to: left cam", out)
        self.assertIn("no face for 50s", out)

    def test_not_visible_with_io_error_detail(self):
        now = 10_000.0
        bc = self._bc(
            last_seen={},
            errors={0: "MSMF read fail"},
            error_at={0: now - 4.0},
        )
        with _patch_bc(bc), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_where_is_user()
        self.assertIn("NOT currently visible", out)
        self.assertIn("never seen user", out)
        self.assertIn("I/O issue", out)
        self.assertIn("MSMF read fail", out)

    def test_recent_but_stale_face(self):
        now = 10_000.0
        # 5s ago -> "saw user 5.0s ago" branch (age < 10).
        bc = self._bc(last_seen={0: now - 5.0, 1: now - 5.0})
        with _patch_bc(bc), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_where_is_user()
        self.assertIn("saw user 5.0s ago", out)
        # neither within 3s -> not visible summary
        self.assertIn("NOT currently visible", out)


# ===========================================================================
# _act_see_screen
# ===========================================================================
class SeeScreenTests(unittest.TestCase):
    def _bc(self, budget_used=0, budget_max=3):
        bc = _base_bc()
        st = types.SimpleNamespace(used=budget_used)
        bc._see_screen_budget_state = st
        bc.SEE_SCREEN_BUDGET_PER_INTENT = budget_max
        # default: no monitor prefix, return question unchanged
        bc._parse_monitor_prefix.side_effect = lambda q: (None, q)
        return bc

    def test_budget_exhausted_refuses(self):
        bc = self._bc(budget_used=3, budget_max=3)
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", {"middle": (0, 0, 100, 100)}):
            out = A._act_see_screen("what's on screen")
        self.assertIn("budget for this intent is exhausted", out)
        bc.take_all_monitor_screenshots.assert_not_called()

    def test_all_monitors_capture(self):
        bc = self._bc()
        bc.take_all_monitor_screenshots.return_value = {"middle": b"PNG"}
        bc.ask_vision_multi.return_value = "a desktop"
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS",
                           {"middle": (0, 0, 100, 100)}):
            out = A._act_see_screen("describe")
        self.assertEqual(out, "a desktop")
        bc.ask_vision_multi.assert_called_once()
        bc._push_screen_context.assert_called_once()
        # budget incremented
        self.assertEqual(bc._see_screen_budget_state.used, 1)

    def test_all_monitors_no_capture(self):
        bc = self._bc()
        bc.take_all_monitor_screenshots.return_value = {}
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", {"middle": (0, 0, 1, 1)}):
            out = A._act_see_screen("")
        self.assertEqual(out, "could not capture any monitor")

    def test_single_monitor_path(self):
        bc = self._bc()
        bc._parse_monitor_prefix.side_effect = lambda q: ("left", "what is here")
        bc.take_screenshot.return_value = b"PNG"
        bc.ask_vision.return_value = "left monitor content"
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS",
                           {"left": (0, 0, 1, 1), "middle": (0, 0, 1, 1)}):
            out = A._act_see_screen("left what is here")
        self.assertEqual(out, "left monitor content")
        bc.take_screenshot.assert_called_once_with(monitor="left")
        bc._push_screen_context.assert_called_once()

    def test_single_monitor_capture_fail(self):
        bc = self._bc()
        bc._parse_monitor_prefix.side_effect = lambda q: ("right", "q")
        bc.take_screenshot.return_value = None
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", {"right": (0, 0, 1, 1)}):
            out = A._act_see_screen("right q")
        self.assertEqual(out, "could not capture screen")

    def test_empty_question_defaults(self):
        # When the parsed question is blank, the default prompt is used.
        bc = self._bc()
        bc._parse_monitor_prefix.side_effect = lambda q: (None, "")
        bc.take_all_monitor_screenshots.return_value = {"m": b"PNG"}
        bc.ask_vision_multi.return_value = "ok"
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", {"m": (0, 0, 1, 1)}):
            A._act_see_screen("")
        q_used = bc.ask_vision_multi.call_args[0][0]
        self.assertIn("Describe in detail", q_used)


# ===========================================================================
# _act_replay_last_action
# ===========================================================================
class ReplayLastActionTests(unittest.TestCase):
    def _bc(self, history, destructive=None):
        bc = _base_bc()
        bc._action_history_lock = mock.MagicMock()
        bc._action_history = history
        bc._DESTRUCTIVE_REPLAY_ACTIONS = set(destructive or
                                             ["close_window", "kill_process"])
        return bc

    def test_no_history(self):
        bc = self._bc([])
        with _patch_bc(bc):
            out = A._act_replay_last_action()
        self.assertEqual(out, "no previous action to replay")

    def test_destructive_refused(self):
        bc = self._bc([{"action": "kill_process", "arg": "notepad"}])
        with _patch_bc(bc):
            out = A._act_replay_last_action()
        self.assertIn("refusing to replay destructive action 'kill_process'",
                      out)

    def test_action_no_longer_registered(self):
        bc = self._bc([{"action": "ghost", "arg": ""}])
        bc.ACTIONS = {}
        with _patch_bc(bc):
            out = A._act_replay_last_action()
        self.assertEqual(out, "cannot replay 'ghost' — action no longer registered")

    def test_replay_success_no_arg(self):
        bc = self._bc([{"action": "screenshot", "arg": ""}])
        fn = mock.Mock(return_value="captured screen")
        bc.ACTIONS = {"screenshot": fn}
        with _patch_bc(bc):
            out = A._act_replay_last_action()
        fn.assert_called_once_with("")
        self.assertEqual(out, "replayed screenshot: captured screen")

    def test_replay_with_monitor_substitution(self):
        bc = self._bc([{"action": "see_screen", "arg": "middle"}])
        fn = mock.Mock(return_value="looked left")
        bc.ACTIONS = {"see_screen": fn}
        bc._substitute_monitor_in_arg.return_value = "left"
        with _patch_bc(bc):
            out = A._act_replay_last_action("left")
        bc._substitute_monitor_in_arg.assert_called_once_with(
            "see_screen", "middle", "left")
        fn.assert_called_once_with("left")
        self.assertIn("on monitor left", out)

    def test_replay_exception(self):
        bc = self._bc([{"action": "boom", "arg": ""}])
        fn = mock.Mock(side_effect=RuntimeError("nope"))
        bc.ACTIONS = {"boom": fn}
        with _patch_bc(bc):
            out = A._act_replay_last_action()
        self.assertEqual(out, "replay of 'boom' failed: nope")

    def test_long_head_truncated(self):
        bc = self._bc([{"action": "see_screen", "arg": ""}])
        fn = mock.Mock(return_value="x" * 300)
        bc.ACTIONS = {"see_screen": fn}
        with _patch_bc(bc):
            out = A._act_replay_last_action()
        self.assertIn("...", out)
        # head capped at ~160 chars + prefix.
        self.assertLessEqual(len(out), 200)

    def test_non_string_result_coerced(self):
        bc = self._bc([{"action": "count", "arg": ""}])
        fn = mock.Mock(return_value=42)
        bc.ACTIONS = {"count": fn}
        with _patch_bc(bc):
            out = A._act_replay_last_action()
        self.assertEqual(out, "replayed count: 42")


# ===========================================================================
# _act_run_shell
# ===========================================================================
class RunShellTests(unittest.TestCase):
    def _bc(self):
        bc = _base_bc()
        bc._SHELL_FORBIDDEN_PATTERNS = ["rm -rf", "format c:"]
        bc.RUN_SHELL_TIMEOUT_SEC = 30
        bc.RUN_SHELL_OUTPUT_MAX_CHARS = 100
        return bc

    def test_empty_command(self):
        bc = self._bc()
        with _patch_bc(bc):
            out = A._act_run_shell("   ")
        self.assertEqual(out, "format: run_shell, <command>")

    def test_blocklisted_command_refused(self):
        bc = self._bc()
        with _patch_bc(bc):
            out = A._act_run_shell("rm -rf /")
        self.assertIn("REFUSED", out)
        self.assertIn("destructive-commands blocklist", out)

    def test_success_output(self):
        bc = self._bc()
        result = types.SimpleNamespace(returncode=0, stdout="hello\n", stderr="")
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run", return_value=result) as run:
            out = A._act_run_shell("echo hello")
        self.assertIn("exit code: 0", out)
        self.assertIn("stdout:\nhello", out)
        # PowerShell invocation shape preserved.
        argv = run.call_args[0][0]
        self.assertEqual(argv[0], "powershell")
        self.assertIn("echo hello", argv)

    def test_stderr_and_exit_code(self):
        bc = self._bc()
        result = types.SimpleNamespace(returncode=1, stdout="", stderr="bad cmd")
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run", return_value=result):
            out = A._act_run_shell("dosomething")
        self.assertIn("exit code: 1", out)
        self.assertIn("stderr:\nbad cmd", out)

    def test_no_output(self):
        bc = self._bc()
        result = types.SimpleNamespace(returncode=0, stdout="  ", stderr="")
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run", return_value=result):
            out = A._act_run_shell("Set-Nothing")
        self.assertIn("(no output)", out)

    def test_output_truncated(self):
        bc = self._bc()
        big = "a" * 250
        result = types.SimpleNamespace(returncode=0, stdout=big, stderr="")
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run", return_value=result):
            out = A._act_run_shell("dump")
        self.assertIn("truncated", out)
        self.assertIn("250 chars total", out)

    def test_timeout(self):
        bc = self._bc()
        exc = A.subprocess.TimeoutExpired(cmd="x", timeout=30)
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run", side_effect=exc):
            out = A._act_run_shell("Start-Sleep 999")
        self.assertIn("timed out after 30s", out)

    def test_powershell_not_found(self):
        bc = self._bc()
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run",
                                  side_effect=FileNotFoundError()):
            out = A._act_run_shell("whatever")
        self.assertIn("powershell.exe not on PATH", out)

    def test_generic_exception(self):
        bc = self._bc()
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run",
                                  side_effect=RuntimeError("weird")):
            out = A._act_run_shell("whatever")
        self.assertTrue(out.startswith("run_shell failed: weird"))

    def test_stderr_truncated(self):
        bc = self._bc()
        big_err = "e" * 250
        result = types.SimpleNamespace(returncode=1, stdout="", stderr=big_err)
        with _patch_bc(bc), \
                mock.patch.object(A.subprocess, "run", return_value=result):
            out = A._act_run_shell("noisy")
        self.assertIn("stderr:", out)
        self.assertIn("truncated", out)
        self.assertIn("250 chars total", out)

    def test_missing_create_no_window_constant_falls_back(self):
        # On the Linux CI host subprocess.CREATE_NO_WINDOW is absent; the
        # handler must swallow the AttributeError and use creationflags=0.
        # Force that path here regardless of host by deleting the attribute
        # (and ensuring os.name looks like Windows so the branch is entered).
        bc = self._bc()
        result = types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
        had = hasattr(A.subprocess, "CREATE_NO_WINDOW")
        saved = getattr(A.subprocess, "CREATE_NO_WINDOW", None)
        if had:
            del A.subprocess.CREATE_NO_WINDOW
        try:
            with _patch_bc(bc), \
                    mock.patch.object(A.os, "name", "nt"), \
                    mock.patch.object(A.subprocess, "run",
                                      return_value=result) as run:
                out = A._act_run_shell("echo ok")
            self.assertIn("exit code: 0", out)
            # creationflags fell back to 0 after the AttributeError.
            self.assertEqual(run.call_args.kwargs["creationflags"], 0)
        finally:
            if had:
                A.subprocess.CREATE_NO_WINDOW = saved


# ===========================================================================
# _act_see_user  (imports cv2 -> injected fake)
# ===========================================================================
class _FakeCv2(types.ModuleType):
    def __init__(self, ok=True):
        super().__init__("cv2")
        self._ok = ok

    def imencode(self, ext, frame):
        class _Buf:
            def tobytes(self):
                return b"PNGDATA"
        return self._ok, _Buf()


class SeeUserTests(unittest.TestCase):
    def _bc(self, **kw):
        bc = _base_bc()
        bc._camera_state_lock = mock.MagicMock()
        bc._camera_last_seen = kw.get("last_seen", {})
        bc._camera_latest_frame = kw.get("latest_frame", {})
        bc._camera_last_frame_at = kw.get("last_frame_at", {})
        bc._camera_last_read_error = kw.get("last_err", {})
        bc.ask_vision.return_value = "a person at a desk"
        return bc

    def _frame(self):
        # Frame stand-in: only needs a .copy() returning itself.
        f = mock.Mock()
        f.copy.return_value = f
        return f

    def test_no_cameras(self):
        bc = self._bc()
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", []):
            out = A._act_see_user()
        self.assertEqual(out, "no cameras configured")

    def test_no_frames_with_last_error(self):
        bc = self._bc(last_err={2: "device busy"})
        cams = [{"index": 2, "label": "cam", "look_x": 0.5}]
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", cams):
            out = A._act_see_user()
        self.assertIn("no webcam frames available yet", out)
        self.assertIn("device busy", out)

    def test_no_frames_no_error(self):
        bc = self._bc()
        cams = [{"index": 0, "label": "cam", "look_x": 0.5}]
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", cams):
            out = A._act_see_user()
        self.assertIn("face tracker may not have started", out)

    def test_describes_user_from_best_frame(self):
        frame = self._frame()
        bc = self._bc(
            last_seen={0: 5000.0},
            latest_frame={0: frame},
            last_frame_at={0: 9999.5},
        )
        cams = [{"index": 0, "label": "cam", "look_x": 0.5}]
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", cams), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_see_user()
        # Frame age ~0.5s -> no stale note appended.
        self.assertEqual(out, "a person at a desk")
        bc.ask_vision.assert_called_once()

    def test_stale_frame_appends_note(self):
        frame = self._frame()
        bc = self._bc(
            last_seen={0: 0.0},
            latest_frame={0: frame},
            last_frame_at={0: 9000.0},   # 1000s old -> stale note
            last_err={0: "timeout"},
        )
        cams = [{"index": 0, "label": "cam", "look_x": 0.5}]
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", cams), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_see_user()
        self.assertIn("a person at a desk", out)
        self.assertIn("frame is", out)
        self.assertIn("last read error: timeout", out)

    def test_encode_failure(self):
        frame = self._frame()
        bc = self._bc(last_seen={0: 5000.0}, latest_frame={0: frame},
                      last_frame_at={0: 9999.0})
        cams = [{"index": 0, "label": "cam", "look_x": 0.5}]
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2(ok=False)}), \
                mock.patch("core.config.CAMERAS", cams), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_see_user()
        self.assertEqual(out, "failed to encode webcam frame")

    def test_frame_dict_has_none_value(self):
        # best_idx resolves via the latest_frame fallback, but the stored
        # frame is None -> "no frame cached for that camera".
        bc = self._bc(last_seen={}, latest_frame={0: None})
        cams = [{"index": 0, "label": "cam", "look_x": 0.5}]
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", cams), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_see_user()
        self.assertEqual(out, "no frame cached for that camera")

    def test_custom_camera_hint_used_as_question(self):
        frame = self._frame()
        bc = self._bc(last_seen={0: 5000.0}, latest_frame={0: frame},
                      last_frame_at={0: 9999.5})
        cams = [{"index": 0, "label": "cam", "look_x": 0.5}]
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", cams), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            A._act_see_user("are they smiling?")
        self.assertEqual(bc.ask_vision.call_args[0][0], "are they smiling?")


# ===========================================================================
# _act_which_monitor  (imports cv2 -> injected fake)
# ===========================================================================
class WhichMonitorTests(unittest.TestCase):
    def _cams(self):
        return [
            {"index": 0, "label": "left", "look_x": 0.2},
            {"index": 1, "label": "right", "look_x": 0.8},
        ]

    def _bc(self, last_seen=None, frames=None):
        bc = _base_bc()
        bc._camera_state_lock = mock.MagicMock()
        bc._camera_last_seen = last_seen or {}
        bc._camera_latest_frame = frames or {}
        return bc

    def test_no_monitors(self):
        bc = self._bc()
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS", {}):
            out = A._act_which_monitor()
        self.assertIn("no MONITORS configured", out)

    def test_not_visible(self):
        bc = self._bc(last_seen={})
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS", {"middle": (0, 0, 1, 1)}), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_which_monitor()
        self.assertIn("not visible to any camera", out)

    def test_left_only(self):
        now = 10_000.0
        bc = self._bc(last_seen={0: now - 1.0})
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS",
                           {"left": (0, 0, 1, 1), "right": (0, 0, 1, 1)}), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_which_monitor()
        self.assertIn("facing LEFT monitor (left)", out)

    def test_right_only(self):
        now = 10_000.0
        bc = self._bc(last_seen={1: now - 1.0})
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS", {"right": (0, 0, 1, 1)}), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_which_monitor()
        self.assertIn("facing RIGHT monitor (right)", out)

    def test_both_no_top_monitor(self):
        now = 10_000.0
        frame = mock.Mock(); frame.copy.return_value = frame
        bc = self._bc(last_seen={0: now - 1.0, 1: now - 1.0},
                      frames={0: frame})
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS", {"middle": (0, 0, 1, 1)}), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_which_monitor()
        self.assertIn("middle/forward (top monitor not configured)", out)

    def test_both_top_configured_vision_says_up(self):
        now = 10_000.0
        frame = mock.Mock(); frame.copy.return_value = frame
        bc = self._bc(last_seen={0: now - 1.0, 1: now - 1.0},
                      frames={0: frame})
        bc.ask_vision.return_value = "UP"
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS",
                           {"top": (0, 0, 1, 1), "middle": (0, 0, 1, 1)}), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_which_monitor()
        self.assertIn("facing TOP monitor (head tilted up)", out)

    def test_both_top_configured_vision_says_forward(self):
        now = 10_000.0
        frame = mock.Mock(); frame.copy.return_value = frame
        bc = self._bc(last_seen={0: now - 1.0, 1: now - 1.0},
                      frames={0: frame})
        bc.ask_vision.return_value = "FORWARD"
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2()}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS",
                           {"top": (0, 0, 1, 1), "middle": (0, 0, 1, 1)}), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_which_monitor()
        self.assertIn("facing MIDDLE monitor (looking forward)", out)

    def test_both_top_configured_but_encode_fails(self):
        now = 10_000.0
        frame = mock.Mock(); frame.copy.return_value = frame
        bc = self._bc(last_seen={0: now - 1.0, 1: now - 1.0},
                      frames={0: frame})
        with _patch_bc(bc), \
                mock.patch.dict(sys.modules, {"cv2": _FakeCv2(ok=False)}), \
                mock.patch("core.config.CAMERAS", self._cams()), \
                mock.patch("core.config.MONITORS",
                           {"top": (0, 0, 1, 1), "middle": (0, 0, 1, 1)}), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_which_monitor()
        self.assertIn("couldn't check head tilt", out)


# ===========================================================================
# _act_session_memory_recall
# ===========================================================================
class SessionMemoryRecallTests(unittest.TestCase):
    def _bc(self):
        bc = _base_bc()
        bc.pattern_memory = mock.Mock()
        return bc

    def test_recall_failure_from_index(self):
        bc = self._bc()
        bc.pattern_memory.get_session_summaries.side_effect = RuntimeError("io")
        with _patch_bc(bc):
            out = A._act_session_memory_recall("what did we do")
        self.assertTrue(out.startswith("session recall failed: io"))

    def test_no_sessions_uses_window_description(self):
        bc = self._bc()
        bc.pattern_memory.get_session_summaries.return_value = []
        bc.pattern_memory.describe_window.return_value = "for yesterday"
        with _patch_bc(bc):
            out = A._act_session_memory_recall("yesterday")
        self.assertIn("no recollection for yesterday", out)

    def test_synthesises_via_llm(self):
        bc = self._bc()
        bc.pattern_memory.get_session_summaries.return_value = [
            {"date": "2026-05-31", "day": "Saturday", "hour_started": 20,
             "hour_ended": 22, "summary": "debugged the TTS pipeline"},
        ]
        bc._llm_quick.return_value = "  Yesterday evening, sir, you debugged TTS.  "
        with _patch_bc(bc):
            out = A._act_session_memory_recall("what did I do last night")
        self.assertEqual(out, "Yesterday evening, sir, you debugged TTS.")
        # The context block fed to the LLM includes the formatted session line.
        user_msg = bc._llm_quick.call_args.kwargs["user"]
        self.assertIn("Saturday 2026-05-31 20:00-22:00", user_msg)
        self.assertIn("debugged the TTS pipeline", user_msg)

    def test_llm_exception(self):
        bc = self._bc()
        bc.pattern_memory.get_session_summaries.return_value = [
            {"date": "2026-05-31", "summary": "stuff"}]
        bc._llm_quick.side_effect = RuntimeError("llm down")
        with _patch_bc(bc):
            out = A._act_session_memory_recall("q")
        self.assertTrue(out.startswith("session recall LLM call failed: llm down"))

    def test_llm_returns_empty(self):
        bc = self._bc()
        bc.pattern_memory.get_session_summaries.return_value = [
            {"date": "2026-05-31", "summary": "stuff"}]
        bc._llm_quick.return_value = "   "
        with _patch_bc(bc):
            out = A._act_session_memory_recall("q")
        self.assertIn("recalled 1 session(s) but the recall LLM returned nothing",
                      out)


# ===========================================================================
# _act_recall_screen
# ===========================================================================
class RecallScreenTests(unittest.TestCase):
    def _bc(self, recent):
        bc = _base_bc()
        bc._recent_screen_contexts.return_value = recent
        bc._format_screen_age.side_effect = lambda age: f"{int(age)}s ago"
        return bc

    def test_nothing_cached(self):
        bc = self._bc([])
        with _patch_bc(bc):
            out = A._act_recall_screen("anything?")
        self.assertIn("haven't seen the screen in the last 5 minutes", out)

    def test_summary_mode(self):
        entry = {"ts": 9_900.0, "monitor": None,
                 "answer": "a terminal and a browser",
                 "images": {"middle": b"PNG"}}
        bc = self._bc([entry])
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_recall_screen("")
        self.assertIn("I last looked at all monitors", out)
        self.assertIn("a terminal and a browser", out)
        bc.ask_vision.assert_not_called()
        bc.ask_vision_multi.assert_not_called()

    def test_summary_mode_truncates_long_snippet(self):
        entry = {"ts": 9_900.0, "monitor": "left",
                 "answer": "z" * 400, "images": {}}
        bc = self._bc([entry])
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_recall_screen("")
        self.assertIn("...", out)

    def test_followup_single_image(self):
        entry = {"ts": 9_950.0, "monitor": "left",
                 "answer": "old answer", "images": {"left": b"PNG"}}
        bc = self._bc([entry])
        bc.ask_vision.return_value = "fresh single-image answer"
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_recall_screen("is the build green?")
        self.assertEqual(out, "fresh single-image answer")
        contextual_q = bc.ask_vision.call_args[0][0]
        self.assertIn("cached screenshot", contextual_q)
        self.assertIn("is the build green?", contextual_q)

    def test_followup_multi_image(self):
        entry = {"ts": 9_950.0, "monitor": None,
                 "answer": "old", "images": {"left": b"P1", "right": b"P2"}}
        bc = self._bc([entry])
        bc.ask_vision_multi.return_value = "multi answer"
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_recall_screen("compare them")
        self.assertEqual(out, "multi answer")
        self.assertIn("cached from", bc.ask_vision_multi.call_args[0][0])

    def test_followup_no_images_cached(self):
        entry = {"ts": 9_950.0, "monitor": "left",
                 "answer": "had text only", "images": {}}
        bc = self._bc([entry])
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=10_000.0):
            out = A._act_recall_screen("what about it")
        self.assertIn("no image to re-examine", out)


# ===========================================================================
# _act_read_changelog
# ===========================================================================
class ReadChangelogTests(unittest.TestCase):
    def _write_changelog(self, td, text):
        with open(os.path.join(td, "CHANGELOG.md"), "w", encoding="utf-8") as f:
            f.write(text)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            bc = _base_bc(td)
            with _patch_bc(bc):
                out = A._act_read_changelog()
        self.assertEqual(out, "I don't have a changelog file yet, sir.")

    def test_no_version_headers(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_changelog(td, "just some prose, no headers")
            bc = _base_bc(td)
            with _patch_bc(bc):
                out = A._act_read_changelog()
        self.assertEqual(out, "The changelog is empty, sir.")

    def test_summarises_latest_entry(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_changelog(
                td,
                "## v1.2.0\n- added voice cloning\n- fixed 3 bugs\n\n"
                "## v1.1.0\n- old stuff\n")
            bc = _base_bc(td)
            bc._llm_quick.return_value = "  Added voice cloning and fixed bugs.  "
            with _patch_bc(bc):
                out = A._act_read_changelog()
        self.assertEqual(out, "Added voice cloning and fixed bugs.")
        # Default: single entry -> 'entry' singular in the system prompt.
        sys_prompt = bc._llm_quick.call_args[0][0]
        self.assertIn("CHANGELOG.md entry", sys_prompt)

    def test_history_request_pulls_three(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_changelog(
                td,
                "## v3\n- c\n\n## v2\n- b\n\n## v1\n- a\n")
            bc = _base_bc(td)
            bc._llm_quick.return_value = "recent summary"
            with _patch_bc(bc):
                out = A._act_read_changelog("what changed lately")
        self.assertEqual(out, "recent summary")
        sys_prompt = bc._llm_quick.call_args[0][0]
        self.assertIn("entries", sys_prompt)  # plural
        user = bc._llm_quick.call_args[0][1]
        self.assertIn("## v3", user)
        self.assertIn("## v1", user)

    def test_long_entry_opens_file(self):
        with tempfile.TemporaryDirectory() as td:
            big = "## v9\n" + ("detail line\n" * 1000)  # > 6000 chars
            self._write_changelog(td, big)
            bc = _base_bc(td)
            with _patch_bc(bc), \
                    mock.patch.object(A.os, "startfile", create=True) as sf:
                out = A._act_read_changelog()
            # On the CI sim sys.platform == 'linux' -> the win32 branch is
            # skipped and the POSIX xdg-open branch runs instead; either way
            # the spoken pointer is returned. Guard both.
            self.assertIn("opened CHANGELOG.md", out)
            del sf  # referenced to keep patch active

    def test_long_entry_posix_xdg_open(self):
        # On a non-win32 host the long-entry branch shells out to xdg-open.
        with tempfile.TemporaryDirectory() as td:
            big = "## v9\n" + ("detail line\n" * 1000)
            self._write_changelog(td, big)
            bc = _base_bc(td)
            with _patch_bc(bc), \
                    mock.patch.object(A.sys, "platform", "linux"), \
                    mock.patch.object(A.subprocess, "Popen") as popen:
                out = A._act_read_changelog()
            popen.assert_called_once()
            self.assertEqual(popen.call_args[0][0][0], "xdg-open")
            self.assertIn("opened CHANGELOG.md", out)

    def test_long_entry_open_failure_swallowed(self):
        # If the opener raises, the except is swallowed and the pointer still
        # returns.
        with tempfile.TemporaryDirectory() as td:
            big = "## v9\n" + ("detail line\n" * 1000)
            self._write_changelog(td, big)
            bc = _base_bc(td)
            with _patch_bc(bc), \
                    mock.patch.object(A.sys, "platform", "linux"), \
                    mock.patch.object(A.subprocess, "Popen",
                                      side_effect=OSError("no xdg")):
                out = A._act_read_changelog()
            self.assertIn("opened CHANGELOG.md", out)

    def test_outer_read_exception(self):
        # open() raising after the existence check hits the outer except.
        with tempfile.TemporaryDirectory() as td:
            self._write_changelog(td, "## v1\n- a\n")
            bc = _base_bc(td)
            with _patch_bc(bc), \
                    mock.patch("builtins.open",
                               side_effect=OSError("read denied")):
                out = A._act_read_changelog()
        self.assertTrue(out.startswith("could not read the changelog: read denied"))

    def test_llm_summary_failure(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_changelog(td, "## v1\n- a\n")
            bc = _base_bc(td)
            bc._llm_quick.side_effect = RuntimeError("llm boom")
            with _patch_bc(bc):
                out = A._act_read_changelog()
        self.assertTrue(out.startswith("could not summarise the changelog: llm boom"))

    def test_llm_empty_summary_falls_back_to_path(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_changelog(td, "## v1\n- a\n")
            bc = _base_bc(td)
            bc._llm_quick.return_value = "   "
            with _patch_bc(bc):
                out = A._act_read_changelog()
        self.assertIn("the full", out)
        self.assertIn("changelog is at", out)


# ===========================================================================
# _act_start_overnight_upgrade
# ===========================================================================
class StartOvernightUpgradeTests(unittest.TestCase):
    def test_arms_engine_and_writes_flag(self):
        with tempfile.TemporaryDirectory() as td:
            flag = os.path.join(td, ".overnight_active")
            bc = _base_bc(td)
            bc._overnight_run_now = mock.Mock()
            bc._sleep_mode = [False]
            bc.OVERNIGHT_FLAG_FILE = flag
            with _patch_bc(bc), \
                    mock.patch.object(A.time, "time", return_value=1000.0):
                out = A._act_start_overnight_upgrade()
            self.assertIn("start generating improvements right away", out)
            bc._overnight_run_now.set.assert_called_once()
            self.assertTrue(bc._sleep_mode[0])
            # Flag file written with the expiry epoch.
            with open(flag, encoding="utf-8") as f:
                self.assertEqual(f.read(), str(1000.0 + 8 * 3600))
            bc._write_hud_state.assert_called_once()

    def test_flag_write_failure_is_swallowed(self):
        bc = _base_bc()
        bc._overnight_run_now = mock.Mock()
        bc._sleep_mode = [False]
        # Point the flag at a path that can't be opened for writing.
        bc.OVERNIGHT_FLAG_FILE = os.path.join(
            tempfile.gettempdir(), "no_such_dir_xyz", "flag")
        with _patch_bc(bc):
            out = A._act_start_overnight_upgrade()
        # Still returns the success line; the engine was armed regardless.
        self.assertIn("start generating improvements", out)
        bc._overnight_run_now.set.assert_called_once()


# ===========================================================================
# _act_open_on_monitor  (imports pygetwindow -> injected fake)
# ===========================================================================
class _FakeWindow:
    def __init__(self, title, width=800, height=600):
        self.title = title
        self.width = width
        self.height = height
        self.moved_to = None
        self.maximized = False
        self.restored = False

    def restore(self):
        self.restored = True

    def moveTo(self, x, y):
        self.moved_to = (x, y)

    def maximize(self):
        self.maximized = True


def _fake_gw(windows, appear_after_snapshot=True):
    """Fake ``pygetwindow`` module. ``getAllWindows`` returns ``windows``.

    With ``appear_after_snapshot`` (the realistic default) the FIRST call —
    which the handler uses to snapshot ``titles_before`` — returns an empty
    list, so the windows count as freshly-opened (``is_new``) on subsequent
    polling calls and the matcher picks them up."""
    mod = types.ModuleType("pygetwindow")
    state = {"calls": 0}

    def _all():
        state["calls"] += 1
        if appear_after_snapshot and state["calls"] == 1:
            return []
        return list(windows)

    mod.getAllWindows = _all
    return mod


def _clock(start=100.0, step=0.1):
    """An auto-incrementing stand-in for ``time.time``: each call returns the
    previous value plus ``step``. Lets the ``_act_open_on_monitor`` polling
    loop run without exhausting a fixed ``side_effect`` list. A small ``step``
    keeps the clock under the 15s deadline so a window that already matches is
    found on the first iteration; a large ``step`` blows past the deadline so
    the no-match path exits promptly."""
    state = {"t": start - step}

    def _now():
        state["t"] += step
        return state["t"]

    return _now


class OpenOnMonitorTests(unittest.TestCase):
    MONS = {"left": (0, 0, 1920, 1080), "right": (1920, 0, 1920, 1080)}

    def _bc(self):
        bc = _base_bc()
        bc._open_url_new_window.return_value = True
        return bc

    def test_bad_format(self):
        bc = self._bc()
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS):
            out = A._act_open_on_monitor("no pipe here")
        self.assertIn("format: open_on_monitor", out)

    def test_unknown_monitor(self):
        bc = self._bc()
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS):
            out = A._act_open_on_monitor("ceiling | example.com")
        self.assertIn("unknown monitor 'ceiling'", out)

    def test_pygetwindow_missing(self):
        bc = self._bc()
        # Make `import pygetwindow` raise ImportError inside the handler.
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": None}):
            out = A._act_open_on_monitor("left | example.com")
        self.assertIn("pygetwindow not available", out)

    def test_opens_url_and_moves_window(self):
        bc = self._bc()
        win = _FakeWindow("Example Domain — Chrome")
        gw = _fake_gw([win])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock()):
            out = A._act_open_on_monitor("left | example.com")
        self.assertIn("opened 'example.com' on left monitor", out)
        self.assertTrue(win.maximized)
        bc._open_url_new_window.assert_called_once_with("example.com")

    def test_url_fallback_to_webbrowser(self):
        bc = self._bc()
        bc._open_url_new_window.return_value = False  # force webbrowser path
        win = _FakeWindow("Example Domain")
        gw = _fake_gw([win])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A.webbrowser, "open") as wb, \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock()):
            A._act_open_on_monitor("left | http://example.com")
        wb.assert_called_once_with("http://example.com")

    def test_app_name_launches_via_sibling(self):
        bc = self._bc()
        win = _FakeWindow("Calculator")
        gw = _fake_gw([win])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A, "_act_launch_app",
                                  return_value="launched") as la, \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock()):
            out = A._act_open_on_monitor("right | calculator")
        la.assert_called_once_with("calculator")
        self.assertIn("on right monitor", out)

    def test_no_matching_window_found(self):
        bc = self._bc()
        gw = _fake_gw([])  # no windows ever appear
        # Big step: deadline = first_call + 15; the second `while time.time()`
        # check returns first_call + 20 > deadline, so the loop exits at once.
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock(step=20.0)):
            out = A._act_open_on_monitor("left | example.com")
        self.assertIn("couldn't find new window to move it", out)

    def test_window_filtering_skips_titleless_tiny_and_erroring(self):
        # Among the candidate windows: one titleless (skip), one tiny (skip),
        # one whose width access raises (the try/except passes), and the real
        # match. Exercises the inner filter branches.
        bc = self._bc()
        titleless = _FakeWindow("")
        tiny = _FakeWindow("Example tiny", width=100, height=100)
        good = _FakeWindow("Example Domain — Chrome", width=800, height=600)

        class _ExplodingWidth:
            # width access raises -> the handler's try/except around the
            # size check swallows it and keeps the window as a candidate.
            title = "Example weird"
            height = 800
            restored = False
            maximized = False
            moved_to = None

            @property
            def width(self):
                raise RuntimeError("no size")

            def restore(self):
                self.restored = True

            def moveTo(self, x, y):
                self.moved_to = (x, y)

            def maximize(self):
                self.maximized = True

        erroring = _ExplodingWidth()
        gw = _fake_gw([titleless, tiny, erroring, good])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock()):
            out = A._act_open_on_monitor("left | example.com")
        # A real window was found and moved despite the noisy candidates.
        self.assertIn("opened 'example.com' on left monitor", out)

    def test_move_failure_reported(self):
        bc = self._bc()
        win = _FakeWindow("Example Domain")
        win.maximize = mock.Mock(side_effect=RuntimeError("no maximize"))
        gw = _fake_gw([win])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock()):
            out = A._act_open_on_monitor("left | example.com")
        self.assertIn("failed to move window", out)


# ===========================================================================
# _act_move_window_to_monitor  (imports win32gui/win32con -> injected fakes)
# ===========================================================================
def _fake_win32():
    win32gui = types.ModuleType("win32gui")
    win32gui.ShowWindow = mock.Mock()
    win32gui.SetWindowPos = mock.Mock()
    win32con = types.ModuleType("win32con")
    win32con.SW_RESTORE = 9
    win32con.SW_MAXIMIZE = 3
    return win32gui, win32con


class MoveWindowToMonitorTests(unittest.TestCase):
    MONS = {"left": (0, 0, 1920, 1080), "right": (1920, 0, 1920, 1080)}

    def _bc(self, matches):
        bc = _base_bc()
        bc._find_windows_by_title.return_value = matches
        return bc

    def test_bad_format(self):
        bc = self._bc([])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS):
            out = A._act_move_window_to_monitor("notepad only")
        self.assertIn("format: move_window_to_monitor", out)

    def test_unknown_monitor(self):
        bc = self._bc([])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS):
            out = A._act_move_window_to_monitor("Notepad | nowhere")
        self.assertIn("unknown monitor 'nowhere'", out)

    def test_empty_title(self):
        bc = self._bc([])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS):
            out = A._act_move_window_to_monitor(" | left")
        self.assertIn("format: move_window_to_monitor", out)

    def test_no_window_match(self):
        bc = self._bc([])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS):
            out = A._act_move_window_to_monitor("Ghost | left")
        self.assertIn("no window matching 'Ghost'", out)

    def test_win32_path_moves_window(self):
        target = mock.Mock()
        target.title = "Notepad"
        target._hWnd = 4242
        bc = self._bc([target])
        win32gui, win32con = _fake_win32()
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules,
                                {"win32gui": win32gui, "win32con": win32con}), \
                mock.patch.object(A.time, "sleep"):
            out = A._act_move_window_to_monitor("Notepad | right")
        self.assertEqual(out, "moved 'Notepad' to right monitor")
        win32gui.SetWindowPos.assert_called_once()
        # Positioned at the right monitor's top-left.
        args = win32gui.SetWindowPos.call_args[0]
        self.assertEqual(args[2], 1920)
        self.assertEqual(args[3], 0)

    def test_win32_path_exception(self):
        target = mock.Mock()
        target.title = "Notepad"
        target._hWnd = 1
        bc = self._bc([target])
        win32gui, win32con = _fake_win32()
        win32gui.SetWindowPos.side_effect = RuntimeError("win32 fail")
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules,
                                {"win32gui": win32gui, "win32con": win32con}), \
                mock.patch.object(A.time, "sleep"):
            out = A._act_move_window_to_monitor("Notepad | right")
        self.assertIn("could not move 'Notepad': win32 fail", out)

    def test_pygetwindow_fallback_when_pywin32_absent(self):
        target = _FakeWindow("Editor")
        target.resizeTo = mock.Mock()
        target.isMaximized = True
        target.restore = mock.Mock()
        bc = self._bc([target])
        # Make `import win32gui` fail -> fall back to pygetwindow high-level API.
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules,
                                {"win32gui": None, "win32con": None}), \
                mock.patch.object(A.time, "sleep"):
            out = A._act_move_window_to_monitor("Editor | left")
        self.assertIn("moved 'Editor' to left monitor (pygetwindow)", out)
        target.restore.assert_called_once()
        target.resizeTo.assert_called_once_with(1920, 1080)

    def test_pygetwindow_fallback_exception(self):
        target = _FakeWindow("Editor")
        target.moveTo = mock.Mock(side_effect=RuntimeError("move boom"))
        bc = self._bc([target])
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", self.MONS), \
                mock.patch.dict(sys.modules,
                                {"win32gui": None, "win32con": None}), \
                mock.patch.object(A.time, "sleep"):
            out = A._act_move_window_to_monitor("Editor | left")
        self.assertIn("could not move 'Editor': move boom", out)


# ===========================================================================
# _act_create_skill
# ===========================================================================
class CreateSkillTests(unittest.TestCase):
    def _bc(self, td):
        bc = _base_bc(td)
        bc.PENDING_SKILLS_DIR = os.path.join(td, "pending_skills")
        return bc

    def test_disabled_when_skills_off_or_not_claude(self):
        with tempfile.TemporaryDirectory() as td:
            bc = self._bc(td)
            with _patch_bc(bc), \
                    mock.patch("core.config.SKILLS_ENABLED", False), \
                    mock.patch("core.config.AI_BACKEND", "claude"):
                out = A._act_create_skill("foo | do a thing")
            self.assertIn("skill creation requires", out)

    def test_bad_format(self):
        with tempfile.TemporaryDirectory() as td:
            bc = self._bc(td)
            with _patch_bc(bc), \
                    mock.patch("core.config.SKILLS_ENABLED", True), \
                    mock.patch("core.config.AI_BACKEND", "claude"):
                out = A._act_create_skill("noseparator")
            self.assertIn("format: create_skill", out)

    def test_writes_valid_skill(self):
        with tempfile.TemporaryDirectory() as td:
            bc = self._bc(td)
            bc._llm_quick.return_value = (
                "```python\n"
                "def register(actions):\n"
                "    actions['hi'] = lambda a: 'hi'\n"
                "```")
            with _patch_bc(bc), \
                    mock.patch("core.config.SKILLS_ENABLED", True), \
                    mock.patch("core.config.AI_BACKEND", "claude"):
                out = A._act_create_skill("My Skill! | greet the user")
            # Name sanitised: lowercased, non-alnum -> underscore.
            self.assertIn("skill 'my_skill_' written to pending_skills/my_skill_.py",
                          out)
            written = os.path.join(td, "pending_skills", "my_skill_.py")
            self.assertTrue(os.path.exists(written))
            with open(written, encoding="utf-8") as f:
                body = f.read()
            # Markdown fences stripped before write.
            self.assertNotIn("```", body)
            self.assertIn("def register(actions):", body)

    def test_blank_name_becomes_unnamed(self):
        # A left-of-pipe that strips to empty -> sanitises to "" -> "unnamed".
        with tempfile.TemporaryDirectory() as td:
            bc = self._bc(td)
            bc._llm_quick.return_value = "def register(actions):\n    pass\n"
            with _patch_bc(bc), \
                    mock.patch("core.config.SKILLS_ENABLED", True), \
                    mock.patch("core.config.AI_BACKEND", "claude"):
                out = A._act_create_skill("   | stuff")
            self.assertIn("pending_skills/unnamed.py", out)

    def test_punctuation_name_sanitised_to_underscores(self):
        # Each non-alnum char becomes '_' (no collapsing) -> "!!!" => "___".
        with tempfile.TemporaryDirectory() as td:
            bc = self._bc(td)
            bc._llm_quick.return_value = "def register(actions):\n    pass\n"
            with _patch_bc(bc), \
                    mock.patch("core.config.SKILLS_ENABLED", True), \
                    mock.patch("core.config.AI_BACKEND", "claude"):
                out = A._act_create_skill("!!! | stuff")
            self.assertIn("pending_skills/___.py", out)

    def test_syntax_error_rejected_nothing_written(self):
        with tempfile.TemporaryDirectory() as td:
            bc = self._bc(td)
            bc._llm_quick.return_value = "def register(actions:\n  pass"
            with _patch_bc(bc), \
                    mock.patch("core.config.SKILLS_ENABLED", True), \
                    mock.patch("core.config.AI_BACKEND", "claude"):
                out = A._act_create_skill("broken | thing")
            self.assertIn("rejected — generated code failed syntax check", out)
            self.assertIn("Nothing was written", out)
            self.assertFalse(
                os.path.exists(os.path.join(td, "pending_skills", "broken.py")))

    def test_llm_generation_failure(self):
        with tempfile.TemporaryDirectory() as td:
            bc = self._bc(td)
            bc._llm_quick.side_effect = RuntimeError("api down")
            with _patch_bc(bc), \
                    mock.patch("core.config.SKILLS_ENABLED", True), \
                    mock.patch("core.config.AI_BACKEND", "claude"):
                out = A._act_create_skill("foo | bar")
            self.assertTrue(out.startswith("skill generation failed: api down"))


# ===========================================================================
# _act_upgrade
# ===========================================================================
class UpgradeTests(unittest.TestCase):
    def test_disabled_by_config(self):
        bc = _base_bc()
        with _patch_bc(bc), \
                mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", False):
            out = A._act_upgrade()
        self.assertIn("Upgrades are disabled, sir", out)

    def test_upgrade_script_not_found(self):
        # __file__ points at a tempdir with no upgrade_jarvis.py up the tree.
        with tempfile.TemporaryDirectory() as td:
            sub = os.path.join(td, "a", "b", "c", "d", "e")
            os.makedirs(sub)
            bc = _base_bc(sub)
            with _patch_bc(bc), \
                    mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True):
                out = A._act_upgrade()
            self.assertIn("upgrade_jarvis.py not found", out)

    def test_empty_queue(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write("# stub\n")
            todo = os.path.join(td, "todo.md")
            with open(todo, "w", encoding="utf-8") as f:
                f.write("- [x] already done\n")  # no open items
            bc = _base_bc(td)
            bc.TODO_FILE = todo
            with _patch_bc(bc), \
                    mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True):
                out = A._act_upgrade()
            self.assertEqual(out, "queue is empty - nothing to upgrade right now")

    def test_spawns_pipeline_and_self_exits(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write("# stub\n")
            todo = os.path.join(td, "todo.md")
            with open(todo, "w", encoding="utf-8") as f:
                f.write("- [ ] task one\n- [ ] task two\n- [x] done\n")
            bc = _base_bc(td)
            bc.TODO_FILE = todo

            # Capture the self-exit thread WITHOUT running its os._exit body.
            class _NoopThread:
                def __init__(self, *a, **k):
                    self._target = k.get("target")
                def start(self):
                    pass

            with _patch_bc(bc), \
                    mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True), \
                    mock.patch.object(A.subprocess, "Popen") as popen, \
                    mock.patch("threading.Thread", _NoopThread):
                out = A._act_upgrade()
            self.assertIn("upgrade initiated for 2 pending task(s)", out)
            popen.assert_called_once()
            # ANTHROPIC_API_KEY scrubbed from the child env.
            env = popen.call_args.kwargs["env"]
            self.assertNotIn("ANTHROPIC_API_KEY", env)

    def test_spawn_failure_reported(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write("# stub\n")
            todo = os.path.join(td, "todo.md")
            with open(todo, "w", encoding="utf-8") as f:
                f.write("- [ ] task\n")
            bc = _base_bc(td)
            bc.TODO_FILE = todo
            with _patch_bc(bc), \
                    mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True), \
                    mock.patch.object(A.subprocess, "Popen",
                                      side_effect=OSError("spawn fail")):
                out = A._act_upgrade()
            self.assertTrue(out.startswith("failed to spawn upgrade: spawn fail"))


if __name__ == "__main__":
    unittest.main()
