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
    # The vision/screenshot actions consult bc.screenshot_privacy_block_reason()
    # before capturing; a bare Mock would return a truthy Mock and trip the
    # privacy refusal. Default it to "not blocked" so handlers run normally —
    # the dedicated privacy tests override this.
    bc.screenshot_privacy_block_reason.return_value = None
    # Same trap, for the LLM backend: handlers that gate on the live brain read
    # bc.AI_BACKEND, and a bare Mock hands back a truthy Mock that equals
    # nothing — so a "requires Claude" gate would refuse even under
    # @mock.patch("core.config.AI_BACKEND", "claude"). Default the fake monolith
    # to the same shipped defaults the real one boots with; tests that care
    # override them. (Actions read the LIVE backend, not core.config's boot
    # value, as of the 2026-07-14 audit — switch_llm only mutates the monolith.)
    bc.AI_BACKEND = "claude"
    bc.CLAUDE_MODEL = "claude-sonnet-5"
    bc.OLLAMA_MODEL = "gemma4:12b"
    bc._get_local_llm_model.return_value = "gemma4:12b"

    # Faithful _percam_side (label-first, look_x<=0.5=left) — the monolith's
    # canonical camera-side rule. A bare Mock would make bc._percam_side a
    # callable that returns a Mock (not "left"/"right"), silently breaking any
    # handler that now routes through it (e.g. _act_which_monitor). 2026-07-14.
    def _percam_side(cam):
        lbl = str(cam.get("label", "")).lower()
        if "left" in lbl:
            return "left"
        if "right" in lbl:
            return "right"
        return "left" if cam.get("look_x", 0.5) <= 0.5 else "right"
    bc._percam_side = _percam_side
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

    def test_version_file_mtime_overrides_stale_pipeline_date(self):
        # last_upgrade_at is written only by the self-upgrade pipeline; a
        # git-checkout release rewrites VERSION but not version.json, so the
        # spoken date went stale (live bug: v1.99.0 announced as "last
        # updated on May 30"). A VERSION file whose mtime is NEWER than
        # last_upgrade_at must win.
        with tempfile.TemporaryDirectory() as td:
            self._write_version(td, {"last_upgrade_at": "2026-05-30T07:03:19"})
            with open(os.path.join(td, "VERSION"), "w",
                      encoding="utf-8") as f:
                f.write("9.9.9\n")   # freshly written → mtime = now
            bc = _base_bc(td)
            with _patch_bc(bc):
                out = A._act_version_info()
            self.assertNotIn("May 30", out)     # stale date replaced
            self.assertIn("this ", out)          # same-day phrasing from mtime

    def test_stale_version_file_keeps_newer_pipeline_date(self):
        # When last_upgrade_at is NEWER than the VERSION mtime (a genuine
        # self-upgrade after the last release deploy), the pipeline date wins.
        from datetime import datetime, timedelta
        with tempfile.TemporaryDirectory() as td:
            future = (datetime.now() + timedelta(days=1)).replace(microsecond=0)
            self._write_version(td, {"last_upgrade_at": future.isoformat()})
            with open(os.path.join(td, "VERSION"), "w",
                      encoding="utf-8") as f:
                f.write("9.9.9\n")
            bc = _base_bc(td)
            with _patch_bc(bc):
                out = A._act_version_info()
            self.assertIn("I'm on version", out)
            # tomorrow's date renders via the >now branch — just pin that the
            # mtime did NOT displace it back to "this ..." same-day phrasing
            # relative to the file write moment ("this" would only appear if
            # mtime won and now==mtime day; future date yields days_ago<0 →
            # weekday phrasing).
            self.assertNotIn("no upgrade timestamp", out)

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
        # Freeze "now"; entries carry a numeric ts (epoch). now = 1_000_000.0,
        # so the last-hour window is (996400.0, now].
        now = 1_000_000.0
        mem = {
            "topics": [
                {"date": "2026-06-01", "ts": now - 1800},   # 30 min ago -> dropped
                {"date": "2026-01-01", "ts": now - 86400},  # a day ago  -> kept
            ],
            "sessions": [
                {"date": "2026-06-01", "ts": now - 600},    # 10 min ago -> dropped
            ],
            "facts": ["durable"],
        }
        bc = self._bc(mem)
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_forget_last_hour()
        self.assertEqual(out, "forgot 2 item(s) from the last hour")
        saved = bc.save_memory.call_args[0][0]
        self.assertEqual(len(saved["topics"]), 1)
        self.assertEqual(saved["sessions"], [])
        # Facts untouched.
        self.assertEqual(saved["facts"], ["durable"])

    def test_thirty_min_dropped_two_hours_kept(self):
        # Regression for the date-only-vs-datetime lexical-compare bug: a
        # same-day entry from 30 min ago MUST be forgotten, while one from 2 h
        # ago MUST survive. The pre-fix code stored date-only ("%Y-%m-%d") and
        # compared lexically against a datetime cutoff, so it could never drop
        # a today entry.
        now = 1_700_000_000.0
        mem = {
            "topics": [
                {"date": "2026-06-07", "ts": now - 1800},   # 30 min ago -> forgotten
                {"date": "2026-06-07", "ts": now - 7200},   # 2 h ago    -> kept
            ],
            "sessions": [],
        }
        bc = self._bc(mem)
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_forget_last_hour()
        self.assertEqual(out, "forgot 1 item(s) from the last hour")
        saved = bc.save_memory.call_args[0][0]
        self.assertEqual([t["ts"] for t in saved["topics"]], [now - 7200])

    def test_legacy_entry_without_ts_is_kept(self):
        # Entries written before the ts field default to ts=0.0 -> treated as
        # old -> never force-forgotten. Non-destructive for pre-fix memory.
        now = 1_700_000_000.0
        mem = {"topics": [{"date": "2020-01-01"}], "sessions": []}
        bc = self._bc(mem)
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=now):
            out = A._act_forget_last_hour()
        self.assertEqual(out, "nothing recent enough to forget")
        bc.save_memory.assert_not_called()

    def test_nothing_recent(self):
        now = 1_700_000_000.0
        mem = {"topics": [{"date": "2020-01-01", "ts": now - 99999}],
               "sessions": []}
        bc = self._bc(mem)
        with _patch_bc(bc), \
                mock.patch.object(A.time, "time", return_value=now):
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
    # The local iTunes library/COM is GONE: play_music now routes to the
    # browser apple_music action (music.apple.com). The dead _play_music_core
    # is never the primary path.
    def test_empty_arg_prompts(self):
        bc = _base_bc()
        with _patch_bc(bc):
            out = A._act_play_music("   ")
        self.assertIn("format: play_music", out)

    def test_routes_to_apple_music(self):
        bc = _base_bc()
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_apple_music",
                                  return_value="streamed via apple music") as am:
            out = A._act_play_music("Smooth Criminal")
        self.assertEqual(out, "streamed via apple music")
        am.assert_called_once_with("Smooth Criminal")
        bc._play_music_core.assert_not_called()

    def test_field_prefix_stripped_before_routing(self):
        bc = _base_bc()
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_apple_music",
                                  return_value="ok") as am:
            A._act_play_music("artist: Michael Jackson")
        am.assert_called_once_with("Michael Jackson")
        bc._play_music_core.assert_not_called()

    def test_song_prefix_stripped(self):
        bc = _base_bc()
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_apple_music", return_value="ok") as am:
            A._act_play_music("song:Earth Song")
        am.assert_called_once_with("Earth Song")

    def test_library_prefix_honest_note_then_streams(self):
        # `library:` used to force the dead local library; now it says so and
        # streams via Apple Music instead. Must NOT call _play_music_core.
        bc = _base_bc()
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_apple_music",
                                  return_value="Queueing Earth Song on Apple Music.") as am:
            out = A._act_play_music("library: Earth Song")
        am.assert_called_once_with("Earth Song")
        self.assertIn("no longer available", out)
        self.assertIn("Queueing Earth Song", out)
        bc._play_music_core.assert_not_called()

    def test_never_calls_dead_com_core(self):
        bc = _base_bc()
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_apple_music", return_value="ok"):
            A._act_play_music("Thriller")
        bc._play_music_core.assert_not_called()


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

    def test_privacy_blocklist_refuses_before_capture(self):
        # A focused window matched SCREENSHOT_PRIVACY_BLOCKLIST: see_screen
        # must speak the refusal and never capture or spend the budget.
        bc = self._bc()
        bc.screenshot_privacy_block_reason.return_value = "1password"
        bc.SCREENSHOT_PRIVACY_REFUSAL = "REFUSED-PRIVATE"
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", {"m": (0, 0, 100, 100)}):
            out = A._act_see_screen("what's on screen")
        self.assertEqual(out, "REFUSED-PRIVATE")
        bc.take_all_monitor_screenshots.assert_not_called()
        bc.take_screenshot.assert_not_called()
        # budget untouched
        self.assertEqual(bc._see_screen_budget_state.used, 0)

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

    def test_pending_count_read_failure_treated_as_zero(self):
        # If opening/reading TODO_FILE raises, the except swallows it and
        # pending stays 0 → "queue is empty" (covers the count except branch).
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write("# stub\n")
            todo = os.path.join(td, "todo.md")
            with open(todo, "w", encoding="utf-8") as f:
                f.write("- [ ] task one\n")
            bc = _base_bc(td)
            bc.TODO_FILE = todo
            real_open = open

            def boom_open(path, *a, **k):
                if os.path.abspath(path) == os.path.abspath(todo):
                    raise OSError("read denied")
                return real_open(path, *a, **k)

            with _patch_bc(bc), \
                    mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True), \
                    mock.patch("builtins.open", side_effect=boom_open):
                out = A._act_upgrade()
            self.assertEqual(out, "queue is empty - nothing to upgrade right now")

    def test_self_exit_closure_calls_os_exit(self):
        # Drive the _self_exit daemon closure: capture the thread target, then
        # run it with time.sleep + os._exit mocked.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write("# stub\n")
            todo = os.path.join(td, "todo.md")
            with open(todo, "w", encoding="utf-8") as f:
                f.write("- [ ] task\n")
            bc = _base_bc(td)
            bc.TODO_FILE = todo
            captured = {}

            def capture_thread(target=None, daemon=None):
                captured["target"] = target
                return mock.Mock()

            with _patch_bc(bc), \
                    mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True), \
                    mock.patch.object(A.subprocess, "Popen"), \
                    mock.patch("threading.Thread", side_effect=capture_thread):
                A._act_upgrade()
            # Now exercise the captured _self_exit body. 2026-07-14: this path
            # used to call a raw os._exit — the LAST one in the tree, and the
            # exact immortal-zombie route v2.0.51/57 removed everywhere else
            # (ExitProcess deadlocks on the loader lock behind a CUDA-parked
            # thread; the process corpses with ~5GB VRAM pinned until reboot).
            # It also skipped the native release, the web socket and the
            # singleton, so the upgrade's relaunched JARVIS met a held :8766
            # and a held singleton. It now runs the hardened teardown.
            bc._hard_exit.side_effect = SystemExit
            with mock.patch.object(A.time, "sleep"), \
                    mock.patch("threading.Timer"), \
                    mock.patch.object(A, "_release_native_resources") as rel, \
                    mock.patch.object(A, "_stop_web_interface_quietly") as web:
                with self.assertRaises(SystemExit):
                    captured["target"]()
            web.assert_called_once()
            rel.assert_called_once()
            bc._release_singleton.assert_called_once()
            # clean=True — upgrade_jarvis.py relaunches JARVIS itself, so the
            # watchdog must NOT also resurrect it (that would double-boot).
            bc._hard_exit.assert_called_once_with(0, clean=True)

    def test_self_exit_closure_exits_even_on_teardown_error(self):
        # If anything in the teardown raises, the closure STILL hard-exits.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "upgrade_jarvis.py"), "w",
                      encoding="utf-8") as f:
                f.write("# stub\n")
            todo = os.path.join(td, "todo.md")
            with open(todo, "w", encoding="utf-8") as f:
                f.write("- [ ] task\n")
            bc = _base_bc(td)
            bc.TODO_FILE = todo
            captured = {}

            def capture_thread(target=None, daemon=None):
                captured["target"] = target
                return mock.Mock()

            with _patch_bc(bc), \
                    mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True), \
                    mock.patch.object(A.subprocess, "Popen"), \
                    mock.patch("threading.Thread", side_effect=capture_thread):
                A._act_upgrade()
            bc._hard_exit.side_effect = SystemExit
            with mock.patch.object(A.time, "sleep",
                                   side_effect=RuntimeError("clock boom")), \
                    mock.patch("threading.Timer"), \
                    mock.patch("builtins.print"):
                with self.assertRaises(SystemExit):
                    captured["target"]()
            bc._hard_exit.assert_called_once_with(0, clean=True)

    def test_upward_search_stops_at_filesystem_root(self):
        # When upgrade_jarvis.py is nowhere up the tree, the loop walks up
        # until parent == search_dir (filesystem root) and breaks → not found.
        # Patch os.path.exists to always say "no upgrade script" and
        # os.path.dirname to bottom out (parent == search_dir) quickly.
        bc = _base_bc()
        real_exists = os.path.exists

        def no_upgrade(path):
            if path.endswith("upgrade_jarvis.py"):
                return False
            return real_exists(path)

        # dirname that bottoms out: returns the same path once at "root".
        seq = ["/level1", "/", "/"]
        it = iter(seq)

        def fake_dirname(p):
            try:
                return next(it)
            except StopIteration:
                return "/"

        with _patch_bc(bc), \
                mock.patch("core.config.OVERNIGHT_UPGRADE_ENABLED", True), \
                mock.patch.object(A.os.path, "exists", side_effect=no_upgrade), \
                mock.patch.object(A.os.path, "dirname", side_effect=fake_dirname):
            out = A._act_upgrade()
        self.assertIn("upgrade_jarvis.py not found", out)


# ===========================================================================
# _act_shutdown_jarvis — graceful full teardown
# ===========================================================================
class ShutdownJarvisTests(unittest.TestCase):
    def _bc(self):
        bc = mock.Mock()
        bc.SHUTDOWN_GOODBYE_LINES = ["Going dark, sir."]
        bc._sleep_mode = [False]
        return bc

    def test_returns_goodbye_and_spawns_thread(self):
        bc = self._bc()
        with _patch_bc(bc), mock.patch("threading.Thread") as mthread:
            out = A._act_shutdown_jarvis("")
        self.assertEqual(out, "Going dark, sir.")
        # sleep_mode flag flipped, goodbye spoken, daemon thread started.
        self.assertTrue(bc._sleep_mode[0])
        bc._speak.assert_called_once()
        mthread.assert_called_once()
        self.assertTrue(mthread.call_args.kwargs.get("daemon"))
        mthread.return_value.start.assert_called_once()

    def test_goodbye_tts_failure_is_swallowed(self):
        bc = self._bc()
        bc._speak.side_effect = RuntimeError("tts down")
        with _patch_bc(bc), mock.patch("threading.Thread"), \
                mock.patch("builtins.print"):
            out = A._act_shutdown_jarvis("")
        self.assertEqual(out, "Going dark, sir.")

    def _capture_target(self, bc):
        """Run _act_shutdown_jarvis with threading.Thread stubbed, returning
        the captured _do_shutdown closure WITHOUT executing it."""
        self._target_bc = bc          # _run_closure asserts on its _hard_exit
        captured = {}

        def capture_thread(target=None, daemon=None, **k):
            captured["target"] = target
            return mock.Mock()

        with _patch_bc(bc), \
                mock.patch("threading.Thread", side_effect=capture_thread):
            A._act_shutdown_jarvis("")
        return captured["target"]

    def _run_closure(self, target, *, saver_alive=False, diag_raises=False):
        """Execute the captured _do_shutdown body with the inner session-saver
        thread stubbed (alive/not), a controllable core.diagnostic_daemons
        (whose stop_diagnostic_daemons optionally raises), and os._exit raising
        SystemExit. Returns the list of printed lines.

        The source does `from core import diagnostic_daemons`, which resolves
        against the `diagnostic_daemons` ATTRIBUTE on the already-imported
        `core` package as well as sys.modules — so we override both and restore
        them, mirroring the smart_home_router injection. (Patching only
        sys.modules let the real, non-raising module win under the CI-sim
        runner, which imports core.diagnostic_daemons for real.)"""
        printed = []

        def inner_thread(target=None, args=(), daemon=None, **k):
            stub = mock.Mock()
            stub.is_alive.return_value = saver_alive
            return stub

        diag_module = types.ModuleType("core.diagnostic_daemons")
        diag_module.stop_diagnostic_daemons = mock.Mock(
            side_effect=RuntimeError("diag boom") if diag_raises else None)

        import core as core_pkg
        old_mod = sys.modules.get("core.diagnostic_daemons")
        had_attr = hasattr(core_pkg, "diagnostic_daemons")
        old_attr = getattr(core_pkg, "diagnostic_daemons", None)

        def restore_diag():
            if old_mod is not None:
                sys.modules["core.diagnostic_daemons"] = old_mod
            else:
                sys.modules.pop("core.diagnostic_daemons", None)
            if had_attr:
                setattr(core_pkg, "diagnostic_daemons", old_attr)
            elif hasattr(core_pkg, "diagnostic_daemons"):
                delattr(core_pkg, "diagnostic_daemons")

        sys.modules["core.diagnostic_daemons"] = diag_module
        setattr(core_pkg, "diagnostic_daemons", diag_module)
        self.addCleanup(restore_diag)

        # 2026-07-12: the closure's finally exits via bc._hard_exit(0,
        # clean=True) — the un-deadlockable TerminateProcess helper — not a
        # raw os._exit (which hung in ExitProcess and left a 22h zombie).
        # SystemExit is raised by the helper double; the failsafe Timer the
        # closure arms first is stubbed so no real 25s thread outlives us.
        self._target_bc._hard_exit.side_effect = SystemExit
        with mock.patch.object(A.time, "sleep"), \
                mock.patch("threading.Timer"), \
                mock.patch("threading.Thread", side_effect=inner_thread), \
                mock.patch("builtins.print",
                           side_effect=lambda *a, **k: printed.append(
                               " ".join(str(x) for x in a))):
            with self.assertRaises(SystemExit):
                target()
        self._target_bc._hard_exit.assert_called_with(0, clean=True)
        return printed

    def test_do_shutdown_closure_runs_full_teardown(self):
        # Every teardown step is a Mock on bc; the closure must reach
        # os._exit(0) (raised as SystemExit) in its finally.
        bc = self._bc()
        bc.load_memory.return_value = {"facts": []}
        target = self._capture_target(bc)
        self._run_closure(target, saver_alive=False)
        # Representative teardown steps were invoked.
        bc.set_state.assert_any_call("sleep")
        bc.save_session_pattern.assert_called_once()
        bc._release_singleton.assert_called_once()

    def test_do_shutdown_closure_tolerates_step_failures(self):
        # EVERY teardown step raises — the per-step try/excepts (including the
        # _face_track_stop / _focus_tracker_stop / hud-kill / diag-daemon
        # branches) swallow them and the closure still reaches os._exit(0).
        bc = self._bc()
        for attr in ("save_session_pattern", "set_state", "close_log",
                     "_restore_prior_power_plan", "_release_singleton"):
            getattr(bc, attr).side_effect = RuntimeError(f"{attr} boom")
        bc.sd.stop.side_effect = RuntimeError("sd boom")
        bc._face_track_stop.set.side_effect = RuntimeError("face boom")
        bc._focus_tracker_stop.set.side_effect = RuntimeError("focus boom")
        # The three hud-kill callables each carry a __name__ and raise.
        for name in ("_shutdown_hud", "_shutdown_tray",
                     "_shutdown_reticle_overlay"):
            killer = getattr(bc, name)
            killer.__name__ = name
            killer.side_effect = RuntimeError(f"{name} boom")
        bc.load_memory.side_effect = RuntimeError("mem boom")
        target = self._capture_target(bc)
        # diag daemons stop_diagnostic_daemons must also raise (diag_raises).
        printed = self._run_closure(target, saver_alive=False, diag_raises=True)
        # Failure-path log lines were emitted (proves the except branches ran).
        joined = "\n".join(printed)
        self.assertIn("diag daemons stop failed", joined)
        self.assertIn("save_session_pattern failed", joined)

    def test_do_shutdown_session_save_timeout_logs(self):
        # The session-saver inner thread stays alive past the join timeout →
        # the "session save timed out" branch is taken, then os._exit fires.
        bc = self._bc()
        bc.load_memory.return_value = {"facts": []}
        target = self._capture_target(bc)
        printed = self._run_closure(target, saver_alive=True)
        self.assertTrue(
            any("session save timed out" in line for line in printed))


# ===========================================================================
# _act_switch_llm — backend switching with allowlist
# ===========================================================================
class SwitchLlmTests(unittest.TestCase):
    def _bc(self, backend="claude", ollama="qwen2.5:14b"):
        bc = mock.Mock()
        bc.AI_BACKEND = backend
        bc.OLLAMA_MODEL = ollama
        bc._KNOWN_OLLAMA_MODELS = {"qwen2.5:14b", "llama3.1:8b"}
        # 2026-07-14 bug-hunt: switch_llm now reports/repoints via the RESOLVED
        # local tag (_get_local_llm_model), not the vestigial OLLAMA_MODEL
        # constant. In these tests the resolved tag IS the configured ollama tag.
        bc._get_local_llm_model.return_value = ollama
        bc._ollama_has_model.return_value = True
        bc._RESOLVED_LOCAL_LLM_MODEL = [ollama]
        return bc

    def test_blank_reports_current_claude(self):
        bc = self._bc(backend="claude")
        with _patch_bc(bc), \
                mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("")
        self.assertIn("current backend: claude", out)
        self.assertIn("claude-x", out)

    def test_blank_reports_current_ollama_model(self):
        bc = self._bc(backend="ollama", ollama="llama3.1:8b")
        with _patch_bc(bc), mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("")
        self.assertIn("current backend: ollama", out)
        self.assertIn("llama3.1:8b", out)

    def test_switch_to_claude(self):
        bc = self._bc(backend="ollama")
        with _patch_bc(bc), mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("claude")
        self.assertEqual(bc.AI_BACKEND, "claude")
        self.assertIn("switched to claude", out)

    def test_anthropic_alias_switches_to_claude(self):
        bc = self._bc(backend="ollama")
        with _patch_bc(bc), mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("anthropic")
        self.assertEqual(bc.AI_BACKEND, "claude")
        self.assertIn("switched to claude", out)

    def test_switch_to_generic_ollama(self):
        bc = self._bc(backend="claude", ollama="qwen2.5:14b")
        with _patch_bc(bc), mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("ollama")
        self.assertEqual(bc.AI_BACKEND, "ollama")
        self.assertIn("switched to ollama", out)
        self.assertIn("qwen2.5:14b", out)

    def test_known_model_tag_switches(self):
        bc = self._bc(backend="claude")
        with _patch_bc(bc), mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("llama3.1:8b")
        self.assertEqual(bc.AI_BACKEND, "ollama")
        self.assertEqual(bc.OLLAMA_MODEL, "llama3.1:8b")
        self.assertIn("switched to ollama / llama3.1:8b", out)

    def test_prefix_recognised_model_tag_switches(self):
        # A tag not in the known set but matching a known family prefix.
        bc = self._bc(backend="claude")
        with _patch_bc(bc), mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("mistral-nemo")
        self.assertEqual(bc.AI_BACKEND, "ollama")
        self.assertEqual(bc.OLLAMA_MODEL, "mistral-nemo")
        self.assertIn("switched to ollama / mistral-nemo", out)

    def test_unknown_tag_rejected(self):
        bc = self._bc(backend="claude")
        with _patch_bc(bc), mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            out = A._act_switch_llm("turbotron9000")
        # backend unchanged on a rejected tag
        self.assertEqual(bc.AI_BACKEND, "claude")
        self.assertIn("unknown backend tag", out)


# ===========================================================================
# _act_find_on_screen — vision target locator
# ===========================================================================
class FindOnScreenTests(unittest.TestCase):
    def _bc(self, coords, monitor=None):
        bc = mock.Mock()
        bc._parse_monitor_prefix.return_value = (monitor, "the play button")
        bc.find_click_target.return_value = coords
        return bc

    def test_found_returns_coords(self):
        bc = self._bc((120, 240))
        with _patch_bc(bc), mock.patch("builtins.print"):
            out = A._act_find_on_screen("the play button")
        self.assertEqual(out, "found at 120,240")

    def test_not_found_returns_message(self):
        bc = self._bc(None)
        with _patch_bc(bc), mock.patch("builtins.print"):
            out = A._act_find_on_screen("the play button")
        self.assertIn("could not find", out)

    def test_monitor_prefix_threaded_through(self):
        bc = self._bc((1, 2), monitor="left")
        with _patch_bc(bc), mock.patch("builtins.print"):
            A._act_find_on_screen("left|the play button")
        # find_click_target is invoked with the parsed monitor.
        _, kwargs = bc.find_click_target.call_args
        self.assertEqual(kwargs.get("monitor"), "left")


# ===========================================================================
# _act_clear_llm_cache + _act_ambient_mode_toggle (small wrappers)
# ===========================================================================
class ClearLlmCacheTests(unittest.TestCase):
    def test_reports_no_cache(self):
        # No bc access needed, but patch _bc anyway to keep the monolith out.
        with _patch_bc(mock.Mock()):
            out = A._act_clear_llm_cache("")
        self.assertIn("no in-process LLM cache to clear", out)


class AmbientModeToggleTests(unittest.TestCase):
    def test_toggles_via_set_with_negated_flag(self):
        # _act_ambient_mode_toggle reads bc._ambient_mode_active[0] and calls
        # _act_ambient_mode_set with the negation. We patch the sibling setter
        # to observe the argument without driving the daemon plumbing.
        bc = mock.Mock()
        bc._ambient_mode_active = [False]
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_ambient_mode_set",
                                  return_value="Ambient mode active, sir.") as mset:
            out = A._act_ambient_mode_toggle("")
        mset.assert_called_once_with(True)     # negation of False
        self.assertEqual(out, "Ambient mode active, sir.")

    def test_toggles_off_when_currently_on(self):
        bc = mock.Mock()
        bc._ambient_mode_active = [True]
        with _patch_bc(bc), \
                mock.patch.object(A, "_act_ambient_mode_set",
                                  return_value="off") as mset:
            A._act_ambient_mode_toggle("")
        mset.assert_called_once_with(False)


# ===========================================================================
# _bc() — the late-bound bobert_companion accessor itself
# ===========================================================================
class BcAccessorTests(unittest.TestCase):
    def test_bc_imports_and_returns_module(self):
        # _bc() does `import bobert_companion; return it`. We inject a fake
        # module so the real ~14K-line monolith is never imported, and assert
        # _bc hands back exactly that object. (The other suites patch _bc out;
        # this one exercises its 2-line body for coverage.)
        fake_mod = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_mod}):
            self.assertIs(A._bc(), fake_mod)


if __name__ == "__main__":
    unittest.main()
