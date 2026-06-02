"""Unit tests for upgrade_jarvis.py — the self-upgrade pipeline (backup,
LLM-driven patching, validation, rollback, changelog/version bump,
orchestration entry points).

SAFETY CONTRACT (this suite NEVER runs a real upgrade):
  * No real git / pip / claude CLI / subprocess work — every subprocess.run /
    subprocess.Popen is mocked.
  * No real network / LLM / threads / sleep.
  * No writes to real source files — PROJECT_DIR and every derived path
    constant is redirected into a per-test tempdir, restored in tearDown.
  * The optional helper modules (blue_green_manager `_bgm`,
    staging_instance `_stg`) are replaced with in-test fakes via
    mock.patch.object so the real deployment plumbing never fires.

ISOLATION: all patches are per-test and auto-restored (mock.patch context
managers / addCleanup). No module-level sys.modules writes. Module path
globals are saved/restored around every test by _UpgBase.

CI-FAITHFUL: upgrade_jarvis imports only stdlib + two local pure-python
helpers, so this suite RUNS (not skips) under tools/run_tests_ci_sim.py.
A handful of tests assert Windows-specific spawn flags; those guard on
sys.platform / attribute presence so they degrade to a softer assertion on
the Linux sim instead of failing.

Bugs found are annotated with `# BUG:` — source is NOT modified.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import upgrade_jarvis as U


# ─────────────────────────── shared fakes ────────────────────────────────


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess."""
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakePopen:
    """Stand-in for subprocess.Popen. Records the argv it was built with and
    never spawns anything."""
    instances: list = []

    def __init__(self, cmd, *a, **k):
        self.cmd = cmd
        self.args = cmd
        self.kwargs = k
        self.pid = 4242
        self._terminated = False
        self._waited = False
        self.returncode = 0
        type(self).instances.append(self)

    def wait(self, timeout=None):
        self._waited = True
        return 0

    def terminate(self):
        self._terminated = True

    def communicate(self, text=None):
        return ("", "")


class _FakeBGM:
    """Configurable fake for blue_green_manager. Records every call so tests
    can assert the handoff choreography without touching real deployment
    state."""
    def __init__(self, tmp):
        self.PROD_LOCK_FILE = os.path.join(tmp, "jarvis.lock")
        self.STAGING_LOCK_FILE = os.path.join(tmp, "jarvis_staging.lock")
        self.calls: list = []
        self._prod_running_seq = [False]   # popped per prod_is_running() call
        self._instances = {}
        self._version = "2.3.4"

    # -- helpers the tests tune --
    def set_prod_running_sequence(self, seq):
        self._prod_running_seq = list(seq)

    def _rec(self, name, **kw):
        self.calls.append((name, kw))

    def names(self):
        return [c[0] for c in self.calls]

    # -- the API surface run_blue_green_handoff() uses --
    def rollback(self, reason="x"):
        self._rec("rollback", reason=reason)
        return {"ok": True}

    def signal_upgrade_aborted(self, reason="x"):
        self._rec("signal_upgrade_aborted", reason=reason)
        return True

    def seed_staging_data(self):
        self._rec("seed_staging_data")

    def read_version(self):
        self._rec("read_version")
        return self._version

    def signal_handoff(self, reason="upgrade", target_version=None, grace_seconds=10):
        self._rec("signal_handoff", reason=reason,
                  target_version=target_version, grace_seconds=grace_seconds)
        return True

    def signal_handoff_failure(self, reason="timeout"):
        self._rec("signal_handoff_failure", reason=reason)
        return True

    def promote_staging(self, new_version=None, staging_pid=None):
        self._rec("promote_staging", new_version=new_version,
                  staging_pid=staging_pid)

    def prod_is_running(self):
        if self._prod_running_seq:
            return self._prod_running_seq.pop(0)
        return False

    def list_instances(self):
        return self._instances


class _FakeSTG:
    """Configurable fake for staging_instance."""
    def __init__(self):
        self.calls: list = []
        self.precompile_result = (True, [])
        self.smoke_result = {"ok": True, "stage_failed": None, "details": {}}

    def precompile_candidate_files(self, files):
        self.calls.append(("precompile", files))
        return self.precompile_result

    def run_smoke_tests(self, candidate_files=None, boot_timeout_s=60.0,
                        reply_timeout_s=90.0):
        self.calls.append(("run_smoke_tests", candidate_files))
        return self.smoke_result


# ───────────────────────────── base class ────────────────────────────────


class _UpgBase(unittest.TestCase):
    """Redirects every module path-constant in upgrade_jarvis into a fresh
    tempdir for the duration of one test, and stubs time.sleep. All overrides
    are restored in tearDown so the real module globals (and the developer's
    real project tree) are untouched."""

    # constants that point at filesystem paths under PROJECT_DIR
    _PATH_ATTRS = (
        "PROJECT_DIR", "TODO_FILE", "SCRIPT", "UPGRADE_LOG",
        "UPGRADE_STREAM_LOG", "BACKUP_DIR", "UPGRADE_SUMMARY_FILE",
        "CHANGELOG_FILE", "VERSION_FILE", "STABILITY_GATES_LOG",
        "GATE_SNAPSHOT_ROOT", "BOOT_SCRIPT", "LOCK_FILE",
        "STABILITY_SMOKE_TOOL",
    )

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)

        # snapshot + redirect path constants
        self._saved = {a: getattr(U, a) for a in self._PATH_ATTRS}
        self.addCleanup(self._restore_consts)

        p = self.tmp
        U.PROJECT_DIR          = p
        U.TODO_FILE            = os.path.join(p, "jarvis_todo.md")
        U.SCRIPT               = os.path.join(p, "bobert_companion.py")
        U.UPGRADE_LOG          = os.path.join(p, "upgrade_log.txt")
        U.UPGRADE_STREAM_LOG   = os.path.join(p, "upgrade_stream.log")
        U.BACKUP_DIR           = os.path.join(p, "backups")
        U.UPGRADE_SUMMARY_FILE = os.path.join(p, ".last_upgrade_summary.json")
        U.CHANGELOG_FILE       = os.path.join(p, "CHANGELOG.md")
        U.VERSION_FILE         = os.path.join(p, "data", "version.json")
        U.STABILITY_GATES_LOG  = os.path.join(p, "data", "stability_gates.jsonl")
        U.GATE_SNAPSHOT_ROOT   = os.path.join(p, "backups")
        U.BOOT_SCRIPT          = os.path.join(p, "_boot_jarvis.ps1")
        U.LOCK_FILE            = os.path.join(p, "jarvis.lock")
        U.STABILITY_SMOKE_TOOL = os.path.join(p, "tools", "stability_smoke_test.py")

        # never really sleep
        self._sleep = mock.patch.object(U.time, "sleep", lambda *a, **k: None)
        self._sleep.start()
        self.addCleanup(self._sleep.stop)

    def _restore_consts(self):
        for a, v in self._saved.items():
            setattr(U, a, v)

    # -- small filesystem helpers --
    def write(self, relpath, text):
        full = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        return full

    def read(self, relpath):
        with open(os.path.join(self.tmp, relpath), encoding="utf-8") as f:
            return f.read()

    def exists(self, relpath):
        return os.path.exists(os.path.join(self.tmp, relpath))


# ════════════════════════ pure helper functions ═══════════════════════════


class GateConfigTests(_UpgBase):
    def test_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("STABILITY_GATE_INTERVAL", "STABILITY_GATE_DURATION_S",
                      "STABILITY_GATE_DISABLE"):
                os.environ.pop(k, None)
            self.assertEqual(U._gate_config(), (10, 300, False))

    def test_custom_values(self):
        with mock.patch.dict(os.environ, {
            "STABILITY_GATE_INTERVAL": "3",
            "STABILITY_GATE_DURATION_S": "45",
            "STABILITY_GATE_DISABLE": "1",
        }):
            self.assertEqual(U._gate_config(), (3, 45, True))

    def test_bad_interval_falls_back(self):
        with mock.patch.dict(os.environ, {"STABILITY_GATE_INTERVAL": "notanint"}):
            self.assertEqual(U._gate_config()[0], 10)

    def test_bad_duration_falls_back(self):
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DURATION_S": "xx"}):
            self.assertEqual(U._gate_config()[1], 300)

    def test_interval_floor_is_one(self):
        with mock.patch.dict(os.environ, {"STABILITY_GATE_INTERVAL": "0"}):
            self.assertEqual(U._gate_config()[0], 1)

    def test_duration_floor_is_thirty(self):
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DURATION_S": "5"}):
            self.assertEqual(U._gate_config()[1], 30)

    def test_disable_requires_exact_one(self):
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": "true"}):
            self.assertFalse(U._gate_config()[2])


class BumpVersionTests(_UpgBase):
    def test_patch_bump_no_features(self):
        self.assertEqual(U._bump_version("1.2.3", False), "1.2.4")

    def test_minor_bump_with_features(self):
        self.assertEqual(U._bump_version("1.2.3", True), "1.3.0")

    def test_minor_bump_zeroes_patch(self):
        self.assertEqual(U._bump_version("4.5.9", True), "4.6.0")

    def test_garbage_version_defaults(self):
        self.assertEqual(U._bump_version("not.a.version", False), "1.0.1")

    def test_extra_segments_ignored(self):
        # only the first three segments are read
        self.assertEqual(U._bump_version("1.2.3.4.5", False), "1.2.4")

    def test_empty_string_defaults(self):
        self.assertEqual(U._bump_version("", True), "1.0.1")


class ClassifyCompletedTests(_UpgBase):
    def test_audit_bucket(self):
        out = U._classify_completed(["Fix [audit-v2] memory leak"])
        self.assertEqual(out["audit"], ["Fix [audit-v2] memory leak"])

    def test_self_diag_bucket(self):
        out = U._classify_completed(["[self-diag] restart watchdog"])
        self.assertEqual(out["self_diag"], ["[self-diag] restart watchdog"])

    def test_research_priority_buckets(self):
        out = U._classify_completed([
            "research-001 add feature",
            "[TOP PRIORITY] do the thing",
            "[HIGH PRIORITY] another",
        ])
        self.assertEqual(len(out["research"]), 3)

    def test_wish_bucket(self):
        out = U._classify_completed(["add a wish feature", "wish-list item"])
        self.assertEqual(len(out["wish"]), 2)

    def test_other_bucket(self):
        out = U._classify_completed(["just a plain task"])
        self.assertEqual(out["other"], ["just a plain task"])

    def test_audit_precedence_over_others(self):
        # "[audit" matches before the wish branch even though " wish" present
        out = U._classify_completed(["[audit] tidy wish handling"])
        self.assertEqual(out["audit"], ["[audit] tidy wish handling"])
        self.assertEqual(out["wish"], [])


class ShortTitleTests(_UpgBase):
    def test_strips_checkbox(self):
        self.assertEqual(U._short_title("- [x] do something"), "do something")

    def test_strips_bold_dates(self):
        self.assertEqual(U._short_title("- [x] **2026-01-01** fix bug"), "fix bug")

    def test_strips_category_tag(self):
        self.assertEqual(U._short_title("- [x] [audit-v2] patch X"), "patch X")

    def test_strips_done_suffix(self):
        self.assertEqual(
            U._short_title("- [x] add foo ✓ DONE — wrote the thing"), "add foo")

    def test_truncates_to_160(self):
        long = "- [x] " + ("z" * 300)
        self.assertEqual(len(U._short_title(long)), 160)

    def test_collapses_double_spaces(self):
        self.assertEqual(U._short_title("- [x] a    b     c"), "a b c")


# ═══════════════════════════ backup_codebase ══════════════════════════════


class BackupCodebaseTests(_UpgBase):
    def _seed_source(self):
        self.write("bobert_companion.py", "print('jarvis')\n")
        self.write("upgrade_jarvis.py", "x = 1\n")
        self.write("jarvis_todo.md", "- [ ] task\n")
        self.write("skills/foo.py", "FOO = 1\n")
        self.write("core/state.py", "S = 0\n")
        self.write("tools/util.py", "U = 2\n")

    def test_creates_timestamped_snapshot(self):
        self._seed_source()
        with mock.patch.object(U.time, "strftime", return_value="2026-01-02_03-04-05"):
            dest = U.backup_codebase()
        self.assertTrue(os.path.isdir(dest))
        self.assertTrue(dest.endswith("2026-01-02_03-04-05"))

    def test_copies_always_files_and_dirs(self):
        self._seed_source()
        dest = U.backup_codebase()
        self.assertTrue(os.path.exists(os.path.join(dest, "bobert_companion.py")))
        self.assertTrue(os.path.exists(os.path.join(dest, "jarvis_todo.md")))
        self.assertTrue(os.path.exists(os.path.join(dest, "skills", "foo.py")))
        self.assertTrue(os.path.exists(os.path.join(dest, "core", "state.py")))
        self.assertTrue(os.path.exists(os.path.join(dest, "tools", "util.py")))

    def test_missing_optional_files_skipped(self):
        # only the todo present — no crash, snapshot still made
        self.write("jarvis_todo.md", "- [ ] x\n")
        dest = U.backup_codebase()
        self.assertTrue(os.path.isdir(dest))
        self.assertFalse(os.path.exists(os.path.join(dest, "bobert_companion.py")))

    def test_prunes_beyond_max_backups(self):
        self._seed_source()
        # pre-create more than MAX_BACKUPS timestamp dirs
        for i in range(U.MAX_BACKUPS + 3):
            ts = f"2020-01-01_00-00-0{i}" if i < 10 else f"2020-01-01_00-00-{i}"
            os.makedirs(os.path.join(U.BACKUP_DIR, ts), exist_ok=True)
        with mock.patch.object(U.time, "strftime", return_value="2030-12-31_23-59-59"):
            U.backup_codebase()
        ts_re = U.re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
        remaining = [d for d in os.listdir(U.BACKUP_DIR) if ts_re.match(d)]
        self.assertEqual(len(remaining), U.MAX_BACKUPS)

    def test_prune_ignores_non_timestamp_dirs(self):
        self._seed_source()
        # a pipeline/ rollback tree and a gate_ snapshot must NOT be pruned
        os.makedirs(os.path.join(U.BACKUP_DIR, "pipeline"), exist_ok=True)
        os.makedirs(os.path.join(U.BACKUP_DIR, "gate_2026_batch1"), exist_ok=True)
        for i in range(U.MAX_BACKUPS + 2):
            os.makedirs(os.path.join(U.BACKUP_DIR, f"2020-01-01_00-00-0{i % 10}"),
                        exist_ok=True)
        with mock.patch.object(U.time, "strftime", return_value="2031-01-01_00-00-00"):
            U.backup_codebase()
        self.assertTrue(os.path.isdir(os.path.join(U.BACKUP_DIR, "pipeline")))
        self.assertTrue(os.path.isdir(os.path.join(U.BACKUP_DIR, "gate_2026_batch1")))


# ═══════════════════════════ get_pending_tasks ════════════════════════════


class GetPendingTasksTests(_UpgBase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(U.get_pending_tasks(), [])

    def test_parses_unchecked_only(self):
        self.write("jarvis_todo.md",
                   "- [ ] first\n- [x] done\n- [ ] second\nplain line\n")
        self.assertEqual(U.get_pending_tasks(), ["first", "second"])

    def test_read_error_returns_empty(self):
        self.write("jarvis_todo.md", "- [ ] x\n")
        with mock.patch("builtins.open", side_effect=OSError("boom")):
            self.assertEqual(U.get_pending_tasks(), [])


# ═══════════════════════════ kill_running_jarvis ══════════════════════════


class KillRunningJarvisTests(_UpgBase):
    def test_kills_enumerated_pids(self):
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            # first call enumerates PIDs, subsequent calls Stop-Process
            if "Get-CimInstance" in " ".join(cmd):
                return _FakeCompleted(stdout="111\n222\n")
            return _FakeCompleted()

        with mock.patch.object(U.subprocess, "run", side_effect=fake_run):
            n = U.kill_running_jarvis()
        self.assertEqual(n, 2)
        # one enumerate + two stop calls
        self.assertEqual(len(calls), 3)

    def test_no_pids_returns_zero(self):
        with mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(stdout="\n")):
            self.assertEqual(U.kill_running_jarvis(), 0)

    def test_enumeration_timeout_returns_zero(self):
        with mock.patch.object(U.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("ps", 15)):
            self.assertEqual(U.kill_running_jarvis(), 0)

    def test_stop_timeout_still_counts(self):
        def fake_run(cmd, *a, **k):
            if "Get-CimInstance" in " ".join(cmd):
                return _FakeCompleted(stdout="333\n")
            raise subprocess.TimeoutExpired("stop", 10)

        with mock.patch.object(U.subprocess, "run", side_effect=fake_run):
            # the kill loop swallows the per-PID timeout; len(pids) still returned
            self.assertEqual(U.kill_running_jarvis(), 1)

    def test_non_digit_lines_filtered(self):
        with mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(stdout="abc\n42\n\n")):
            with mock.patch.object(U.subprocess, "run") as run2:
                run2.return_value = _FakeCompleted(stdout="abc\n42\n\n")
                # second run for Stop-Process
                n = U.kill_running_jarvis()
        self.assertEqual(n, 1)


# ═══════════════════════ stability-gate helpers ═══════════════════════════


class LogGateEventTests(_UpgBase):
    def test_appends_jsonl_with_ts(self):
        U._log_gate_event({"batch": 1, "verdict": "PASS"})
        text = self.read(os.path.join("data", "stability_gates.jsonl"))
        rec = json.loads(text.strip())
        self.assertEqual(rec["batch"], 1)
        self.assertIn("ts", rec)

    def test_never_raises_on_error(self):
        with mock.patch("builtins.open", side_effect=OSError("nope")):
            # must swallow
            U._log_gate_event({"x": 1})

    def test_preserves_explicit_ts(self):
        U._log_gate_event({"verdict": "FAIL", "ts": "2020-01-01T00:00:00"})
        rec = json.loads(self.read(os.path.join("data", "stability_gates.jsonl")).strip())
        self.assertEqual(rec["ts"], "2020-01-01T00:00:00")


class GateSnapshotTests(_UpgBase):
    def test_copies_surface(self):
        self.write("bobert_companion.py", "x=1\n")
        self.write("skills/a.py", "a=1\n")
        self.write("core/b.py", "b=1\n")
        with mock.patch.object(U.time, "strftime", return_value="2026-01-01_00-00-00"):
            dest = U._gate_snapshot(7)
        self.assertIn("gate_2026-01-01_00-00-00_batch7", dest)
        self.assertTrue(os.path.exists(os.path.join(dest, "bobert_companion.py")))
        self.assertTrue(os.path.exists(os.path.join(dest, "skills", "a.py")))
        self.assertTrue(os.path.exists(os.path.join(dest, "core", "b.py")))

    def test_copy_oserror_swallowed(self):
        self.write("bobert_companion.py", "x=1\n")
        with mock.patch.object(U.shutil, "copy2", side_effect=OSError("locked")):
            # must not raise; dir still made
            dest = U._gate_snapshot(1)
        self.assertTrue(os.path.isdir(dest))


class EventlogAppcrashDumpTests(_UpgBase):
    def test_returns_stdout_stripped(self):
        with mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(stdout="  {\"a\":1}  \n")):
            self.assertEqual(U._eventlog_appcrash_dump("2026-01-01T00:00:00"),
                             '{"a":1}')

    def test_subprocess_error_returns_empty(self):
        with mock.patch.object(U.subprocess, "run",
                               side_effect=OSError("no powershell")):
            self.assertEqual(U._eventlog_appcrash_dump("2026-01-01T00:00:00"), "")

    def test_timeout_returns_empty(self):
        with mock.patch.object(U.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("ps", 30)):
            self.assertEqual(U._eventlog_appcrash_dump("2026-01-01T00:00:00"), "")


class LatestSessionLogTailTests(_UpgBase):
    def test_no_logs_dir_returns_empty(self):
        self.assertEqual(U._latest_session_log_tail(), [])

    def test_no_matching_files_returns_empty(self):
        self.write("logs/other.txt", "hi\n")
        self.assertEqual(U._latest_session_log_tail(), [])

    def test_returns_tail_of_freshest(self):
        self.write("logs/session_old.log", "old\n")
        newp = self.write("logs/session_new.log",
                          "\n".join(f"line{i}" for i in range(10)) + "\n")
        # make "new" the freshest
        os.utime(os.path.join(self.tmp, "logs", "session_old.log"),
                 (1000, 1000))
        os.utime(newp, (2000, 2000))
        tail = U._latest_session_log_tail(n=3)
        self.assertEqual(tail, ["line7", "line8", "line9"])

    def test_unreadable_file_returns_empty(self):
        self.write("logs/session_x.log", "data\n")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(U._latest_session_log_tail(), [])


# ═══════════════════ _run_stability_smoke_test ════════════════════════════


class RunStabilitySmokeTestTests(_UpgBase):
    def _make_tool(self):
        self.write(os.path.join("tools", "stability_smoke_test.py"), "# tool\n")

    def test_missing_tool_returns_error(self):
        out = U._run_stability_smoke_test(duration_s=1)
        self.assertFalse(out["ok"])
        self.assertIn("missing", out["error"])

    def test_pass_report_loaded(self):
        self._make_tool()
        self.write("stability_smoke_PASS.json", json.dumps({"verdict": "PASS"}))
        with mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=0, stdout="ok")):
            out = U._run_stability_smoke_test(duration_s=1)
        self.assertTrue(out["ok"])
        self.assertEqual(out["rc"], 0)
        self.assertEqual(out["report"], {"verdict": "PASS"})

    def test_fail_rc_nonzero(self):
        self._make_tool()
        with mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=3, stdout="bad")):
            out = U._run_stability_smoke_test(duration_s=1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["rc"], 3)

    def test_timeout_returns_error(self):
        self._make_tool()
        exc = subprocess.TimeoutExpired("smoke", 5)
        exc.stdout = "partial-output"
        with mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "run", side_effect=exc):
            out = U._run_stability_smoke_test(duration_s=1)
        self.assertFalse(out["ok"])
        self.assertIn("timed out", out["error"])
        self.assertIn("partial-output", out["stdout_tail"])

    def test_spawn_oserror_returns_error(self):
        self._make_tool()
        with mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "run", side_effect=OSError("denied")):
            out = U._run_stability_smoke_test(duration_s=1)
        self.assertFalse(out["ok"])
        self.assertIn("could not spawn", out["error"])

    def test_duration_none_reads_env(self):
        self._make_tool()
        captured = {}

        def fake_run(cmd, *a, **k):
            captured["cmd"] = cmd
            return _FakeCompleted(returncode=0)

        with mock.patch.dict(os.environ, {"STABILITY_GATE_DURATION_S": "77"}), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "run", side_effect=fake_run):
            U._run_stability_smoke_test(duration_s=None)
        self.assertIn("77", captured["cmd"])

    def test_corrupt_report_file_skipped(self):
        # report file exists but is invalid JSON -> the read except (continue)
        # fires and report stays None; result still reflects the rc.
        self._make_tool()
        self.write("stability_smoke_PASS.json", "{not json")
        with mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=0)):
            out = U._run_stability_smoke_test(duration_s=1)
        self.assertTrue(out["ok"])
        self.assertIsNone(out["report"])

    def test_lock_file_removed_before_run(self):
        self._make_tool()
        lock = self.write("jarvis.lock", "pid")
        self.assertTrue(os.path.exists(lock))
        with mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=0)):
            U._run_stability_smoke_test(duration_s=1)
        self.assertFalse(os.path.exists(lock))


# ═══════════════════════ _revert_to_snapshot ══════════════════════════════


class RevertToSnapshotTests(_UpgBase):
    def test_missing_snapshot_dir(self):
        ok, log = U._revert_to_snapshot(os.path.join(self.tmp, "nope"))
        self.assertFalse(ok)
        self.assertIn("missing", log)

    def test_restores_always_files(self):
        snap = os.path.join(self.tmp, "snap")
        os.makedirs(snap)
        with open(os.path.join(snap, "bobert_companion.py"), "w") as f:
            f.write("GOOD = 1\n")
        # current (broken) file
        self.write("bobert_companion.py", "BROKEN(\n")
        ok, _log = U._revert_to_snapshot(snap)
        self.assertTrue(ok)
        self.assertEqual(self.read("bobert_companion.py"), "GOOD = 1\n")

    def test_copy_failure_marks_not_ok(self):
        snap = os.path.join(self.tmp, "snap")
        os.makedirs(snap)
        with open(os.path.join(snap, "upgrade_jarvis.py"), "w") as f:
            f.write("x=1\n")
        with mock.patch.object(U.shutil, "copy2", side_effect=OSError("ro")):
            ok, log = U._revert_to_snapshot(snap)
        self.assertFalse(ok)
        self.assertIn("failed", log)

    def test_subdir_robocopy_and_purge(self):
        snap = os.path.join(self.tmp, "snap")
        os.makedirs(os.path.join(snap, "core"))
        with open(os.path.join(snap, "core", "keep.py"), "w") as f:
            f.write("k=1\n")
        # destination has an extra file not in the snapshot manifest
        self.write("core/keep.py", "k=0\n")
        self.write("core/extra.py", "e=1\n")

        captured = {}

        def fake_run(cmd, *a, **k):
            captured["cmd"] = cmd
            return _FakeCompleted(returncode=1)   # 1 = success for robocopy

        with mock.patch.object(U.subprocess, "run", side_effect=fake_run):
            ok, log = U._revert_to_snapshot(snap)
        self.assertTrue(ok)
        self.assertEqual(captured["cmd"][0], "robocopy")
        # extra.py purged because it's absent from the manifest
        self.assertFalse(self.exists("core/extra.py"))
        self.assertIn("purged", log)

    def test_robocopy_failure_code_8(self):
        snap = os.path.join(self.tmp, "snap")
        os.makedirs(os.path.join(snap, "tools"))
        with open(os.path.join(snap, "tools", "x.py"), "w") as f:
            f.write("x=1\n")
        self.write("tools/x.py", "x=0\n")
        with mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=8)):
            ok, log = U._revert_to_snapshot(snap)
        self.assertFalse(ok)

    def test_robocopy_oserror(self):
        snap = os.path.join(self.tmp, "snap")
        os.makedirs(os.path.join(snap, "hud"))
        with open(os.path.join(snap, "hud", "y.py"), "w") as f:
            f.write("y=1\n")
        self.write("hud/y.py", "y=0\n")
        with mock.patch.object(U.subprocess, "run", side_effect=OSError("boom")):
            ok, log = U._revert_to_snapshot(snap)
        self.assertFalse(ok)
        self.assertIn("robocopy", log)

    def test_pycache_skipped_in_purge(self):
        snap = os.path.join(self.tmp, "snap")
        os.makedirs(os.path.join(snap, "core"))
        with open(os.path.join(snap, "core", "keep.py"), "w") as f:
            f.write("k=1\n")
        self.write("core/keep.py", "k=1\n")
        self.write("core/__pycache__/keep.cpython.pyc", "bytecode")
        with mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=0)):
            U._revert_to_snapshot(snap)
        # __pycache__ artifact left intact (not purged)
        self.assertTrue(self.exists("core/__pycache__/keep.cpython.pyc"))


# ═══════════════════════════ _stability_gate ══════════════════════════════


class StabilityGateTests(_UpgBase):
    def test_disabled_returns_skip(self):
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": "1"}):
            out = U._stability_gate(1, ["t"], batch_size=2)
        self.assertEqual(out["verdict"], "SKIP")
        self.assertTrue(out["ok"])

    def test_pass_path(self):
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": "",
                                          "STABILITY_GATE_DURATION_S": "30"}), \
             mock.patch.object(U, "_gate_snapshot",
                               return_value=os.path.join(self.tmp, "gate_x")), \
             mock.patch.object(U, "_run_stability_smoke_test",
                               return_value={"ok": True, "report": {"v": "PASS"}}), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0):
            os.makedirs(os.path.join(self.tmp, "gate_x"), exist_ok=True)
            out = U._stability_gate(2, ["task a", "task b"], batch_size=2)
        self.assertEqual(out["verdict"], "PASS")
        self.assertTrue(out["ok"])
        # a PASS marker file was written into the snapshot dir
        markers = [f for f in os.listdir(os.path.join(self.tmp, "gate_x"))
                   if f.startswith("PASS_batch_2")]
        self.assertEqual(len(markers), 1)

    def test_fail_path_reverts_and_queues_regression(self):
        self.write("jarvis_todo.md", "- [ ] existing\n")
        gate_dir = os.path.join(self.tmp, "gate_y")
        os.makedirs(gate_dir, exist_ok=True)
        smoke = {
            "ok": False, "error": "boot crashed",
            "rc": 2, "stdout_tail": "trace...",
            "report": {"checks": {"session_log_fatal": {
                "fatal_lines": ["[FATAL] kaboom"]}}},
        }
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": ""}), \
             mock.patch.object(U, "_gate_snapshot", return_value=gate_dir), \
             mock.patch.object(U, "_run_stability_smoke_test", return_value=smoke), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "_eventlog_appcrash_dump", return_value="{}"), \
             mock.patch.object(U, "_latest_session_log_tail", return_value=["x"]), \
             mock.patch.object(U, "_revert_to_snapshot",
                               return_value=(True, "reverted")) as revert:
            out = U._stability_gate(3, ["broke it"], batch_size=1)
        self.assertEqual(out["verdict"], "FAIL")
        self.assertFalse(out["ok"])
        revert.assert_called_once_with(gate_dir)
        todo = self.read("jarvis_todo.md")
        self.assertIn("[regression]", todo)
        self.assertIn("boot crashed", todo)

    def test_fail_context_chunks_all_categories(self):
        # symptom picked from fatal_lines, but a crash signature + transcribe +
        # skill + milestone findings also present -> every context_chunk branch
        # fires and is appended to the regression line.
        gate_dir = os.path.join(self.tmp, "gate_ctx")
        os.makedirs(gate_dir, exist_ok=True)
        self.write("jarvis_todo.md", "")
        smoke = {
            "ok": False, "error": None, "rc": 1, "stdout_tail": "",
            "report": {"checks": {
                "session_log_fatal": {
                    "fatal_lines": ["[FATAL] primary symptom"],
                    "transcribe_failures": ["whisper died"],
                    "skill_failures": ["weather skill 500", "calendar skill 401"],
                    "boot_milestones": {"mic_ready": True, "tts_ready": False},
                },
                "crash_traces": {"head_signatures": ["numpy SIGSEGV"]},
            }},
        }
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": ""}), \
             mock.patch.object(U, "_gate_snapshot", return_value=gate_dir), \
             mock.patch.object(U, "_run_stability_smoke_test", return_value=smoke), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "_eventlog_appcrash_dump", return_value=""), \
             mock.patch.object(U, "_latest_session_log_tail", return_value=[]), \
             mock.patch.object(U, "_revert_to_snapshot", return_value=(True, "x")):
            U._stability_gate(9, ["a", "b"], batch_size=2)
        todo = self.read("jarvis_todo.md")
        self.assertIn("native crash signature: numpy SIGSEGV", todo)
        self.assertIn("transcribe failures: whisper died", todo)
        self.assertIn("skill failures:", todo)

    def test_fail_symptom_boot_milestones(self):
        # no fatal/traceback/crash -> symptom falls through to missed milestones
        gate_dir = os.path.join(self.tmp, "gate_ms")
        os.makedirs(gate_dir, exist_ok=True)
        self.write("jarvis_todo.md", "")
        smoke = {
            "ok": False, "error": None, "rc": 1, "stdout_tail": "",
            "report": {"checks": {"session_log_fatal": {
                "boot_milestones": {"mic_ready": False, "tts_ready": False}}}},
        }
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": ""}), \
             mock.patch.object(U, "_gate_snapshot", return_value=gate_dir), \
             mock.patch.object(U, "_run_stability_smoke_test", return_value=smoke), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "_eventlog_appcrash_dump", return_value=""), \
             mock.patch.object(U, "_latest_session_log_tail", return_value=[]), \
             mock.patch.object(U, "_revert_to_snapshot", return_value=(True, "x")):
            out = U._stability_gate(10, ["t"], batch_size=1)
        self.assertIn("boot stalled", out["details"])

    def test_fail_regression_append_oserror_swallowed(self):
        # the regression-task append to TODO_FILE raising OSError must not
        # propagate out of the gate.
        gate_dir = os.path.join(self.tmp, "gate_ap")
        os.makedirs(gate_dir, exist_ok=True)
        smoke = {"ok": False, "error": "boom", "rc": 1, "stdout_tail": "",
                 "report": None}
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": ""}), \
             mock.patch.object(U, "_gate_snapshot", return_value=gate_dir), \
             mock.patch.object(U, "_run_stability_smoke_test", return_value=smoke), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "_eventlog_appcrash_dump", return_value=""), \
             mock.patch.object(U, "_latest_session_log_tail", return_value=[]), \
             mock.patch.object(U, "_revert_to_snapshot", return_value=(True, "x")), \
             mock.patch("builtins.open", side_effect=OSError("readonly todo")):
            out = U._stability_gate(11, ["t"], batch_size=1)
        self.assertEqual(out["verdict"], "FAIL")

    def test_fail_symptom_prefers_native_crash(self):
        gate_dir = os.path.join(self.tmp, "gate_z")
        os.makedirs(gate_dir, exist_ok=True)
        self.write("jarvis_todo.md", "")
        smoke = {
            "ok": False, "error": None, "rc": 1, "stdout_tail": "",
            "report": {"checks": {
                "session_log_fatal": {},
                "crash_traces": {"head_signatures": ["sounddevice SIGSEGV"]},
            }},
        }
        with mock.patch.dict(os.environ, {"STABILITY_GATE_DISABLE": ""}), \
             mock.patch.object(U, "_gate_snapshot", return_value=gate_dir), \
             mock.patch.object(U, "_run_stability_smoke_test", return_value=smoke), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "_eventlog_appcrash_dump", return_value=""), \
             mock.patch.object(U, "_latest_session_log_tail", return_value=[]), \
             mock.patch.object(U, "_revert_to_snapshot", return_value=(False, "x")):
            out = U._stability_gate(4, ["t"], batch_size=1)
        self.assertIn("native crash", out["details"])


# ═══════════════════════════ find_claude_cli ══════════════════════════════


class FindClaudeCliTests(_UpgBase):
    def test_uses_which_first(self):
        with mock.patch.object(U.shutil, "which", return_value=r"C:\bin\claude.exe"):
            self.assertEqual(U.find_claude_cli(), r"C:\bin\claude.exe")

    def test_falls_back_to_known_path(self):
        target = os.path.join(self.tmp, "claude.exe")
        with open(target, "w") as f:
            f.write("")
        with mock.patch.object(U.shutil, "which", return_value=None), \
             mock.patch.object(U.os.path, "expanduser", return_value=target), \
             mock.patch.object(U.os.path, "expandvars", return_value=r"C:\missing.exe"):
            self.assertEqual(U.find_claude_cli(), target)

    def test_returns_none_when_nothing_found(self):
        with mock.patch.object(U.shutil, "which", return_value=None), \
             mock.patch.object(U.os.path, "expanduser",
                               return_value=r"C:\nope1.exe"), \
             mock.patch.object(U.os.path, "expandvars",
                               return_value=r"C:\nope2.exe"), \
             mock.patch.object(U.os.path, "exists", return_value=False):
            self.assertIsNone(U.find_claude_cli())


# ═══════════════════════════ spawn_claude_code ════════════════════════════


class SpawnClaudeCodeTests(_UpgBase):
    def setUp(self):
        super().setUp()
        _FakePopen.instances = []

    def _spawn(self, single_stage=False):
        with mock.patch.object(U.subprocess, "Popen", _FakePopen):
            return U.spawn_claude_code(["task one", "task two"],
                                       r"C:\bin\claude.exe",
                                       single_stage=single_stage)

    def test_returns_popen_handle(self):
        proc = self._spawn()
        self.assertIsInstance(proc, _FakePopen)

    def test_renders_pipeline_template_by_default(self):
        self._spawn(single_stage=False)
        # driver tempfile was written and passed to Popen as argv[1]
        argv = _FakePopen.instances[0].cmd
        drv = argv[2]
        with open(drv, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("run_pipeline_loop_driver", src)
        os.unlink(drv)

    def test_renders_single_stage_template(self):
        self._spawn(single_stage=True)
        drv = _FakePopen.instances[0].cmd[2]
        with open(drv, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("run_one_claude_call", src)
        # sentinel substitution happened — no raw sentinels remain
        self.assertNotIn("__PROJECT_DIR__", src)
        self.assertNotIn("__MAX_ITER__", src)
        os.unlink(drv)

    def test_strips_api_key_from_env(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}):
            self._spawn()
        env = _FakePopen.instances[0].kwargs["env"]
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        drv = _FakePopen.instances[0].cmd[2]
        if os.path.exists(drv):
            os.unlink(drv)

    def test_writes_upgrade_log_line(self):
        self._spawn()
        self.assertTrue(self.exists("upgrade_log.txt"))
        self.assertIn("Spawning Claude Code LOOP", self.read("upgrade_log.txt"))
        drv = _FakePopen.instances[0].cmd[2]
        if os.path.exists(drv):
            os.unlink(drv)

    def test_popen_exception_returns_none(self):
        with mock.patch.object(U.subprocess, "Popen",
                               side_effect=OSError("spawn fail")):
            out = U.spawn_claude_code(["t"], r"C:\bin\claude.exe")
        self.assertIsNone(out)


# ═════════════════════════ spawn_claude_dry_audit ═════════════════════════


class SpawnClaudeDryAuditTests(_UpgBase):
    def setUp(self):
        super().setUp()
        _FakePopen.instances = []

    def test_returns_handle_and_logs(self):
        with mock.patch.object(U.subprocess, "Popen", _FakePopen):
            proc = U.spawn_claude_dry_audit(["t1"], r"C:\bin\claude.exe")
        self.assertIsInstance(proc, _FakePopen)
        self.assertIn("DRY AUDIT", self.read("upgrade_log.txt"))

    def test_strips_api_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}), \
             mock.patch.object(U.subprocess, "Popen", _FakePopen):
            U.spawn_claude_dry_audit(["t"], r"C:\bin\claude.exe")
        env = _FakePopen.instances[0].kwargs["env"]
        self.assertNotIn("ANTHROPIC_API_KEY", env)

    def test_quotes_escaped_in_prompt(self):
        # the PS command doubles single quotes in the prompt
        with mock.patch.object(U.subprocess, "Popen", _FakePopen):
            U.spawn_claude_dry_audit(["t"], r"C:\bin\claude.exe")
        ps_cmd = _FakePopen.instances[0].cmd[-1]
        self.assertIn("Tee-Object", ps_cmd)

    def test_exception_returns_none(self):
        with mock.patch.object(U.subprocess, "Popen", side_effect=OSError("x")):
            self.assertIsNone(
                U.spawn_claude_dry_audit(["t"], r"C:\bin\claude.exe"))


# ═══════════════════════════ relaunch_jarvis ══════════════════════════════


class RelaunchJarvisTests(_UpgBase):
    def setUp(self):
        super().setUp()
        _FakePopen.instances = []

    def test_basic_relaunch(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(U.subprocess, "Popen", _FakePopen):
            os.environ.pop("JARVIS_AMBIENT_LEARNING", None)
            os.environ.pop("JARVIS_WAKE_RESUME", None)
            U.relaunch_jarvis()
        ps = _FakePopen.instances[0].cmd[-1]
        self.assertIn("bobert_companion.py", ps)
        self.assertNotIn("--resume-handoff", ps)

    def test_handoff_adds_resume_flag(self):
        with mock.patch.object(U.subprocess, "Popen", _FakePopen):
            U.relaunch_jarvis(with_handoff=True)
        ps = _FakePopen.instances[0].cmd[-1]
        self.assertIn("--resume-handoff", ps)

    def test_ambient_env_propagated(self):
        with mock.patch.dict(os.environ, {"JARVIS_AMBIENT_LEARNING": "0",
                                          "JARVIS_WAKE_RESUME": "stay_talkative"}), \
             mock.patch.object(U.subprocess, "Popen", _FakePopen):
            U.relaunch_jarvis()
        ps = _FakePopen.instances[0].cmd[-1]
        self.assertIn("'0'", ps)
        self.assertIn("stay_talkative", ps)


# ═══════════════════════ spawn_staging_jarvis ═════════════════════════════


class SpawnStagingJarvisTests(_UpgBase):
    def setUp(self):
        super().setUp()
        _FakePopen.instances = []

    def test_sets_staging_env_and_argv(self):
        with mock.patch.object(U.subprocess, "Popen", _FakePopen), \
             mock.patch.object(U.os.path, "exists", return_value=False):
            proc = U.spawn_staging_jarvis()
        self.assertIsInstance(proc, _FakePopen)
        inst = _FakePopen.instances[0]
        self.assertIn("--staging", inst.cmd)
        self.assertEqual(inst.kwargs["env"]["JARVIS_STAGING"], "1")

    def test_uses_pythonw_when_present(self):
        fake_pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        with mock.patch.object(U.subprocess, "Popen", _FakePopen), \
             mock.patch.object(U.os.path, "exists",
                               side_effect=lambda p: p == fake_pythonw):
            U.spawn_staging_jarvis()
        self.assertEqual(_FakePopen.instances[0].cmd[0], fake_pythonw)

    def test_oserror_returns_none(self):
        with mock.patch.object(U.subprocess, "Popen", side_effect=OSError("x")), \
             mock.patch.object(U.os.path, "exists", return_value=False):
            self.assertIsNone(U.spawn_staging_jarvis())


# ═══════════════════ blue-green small helpers ═════════════════════════════


class ProdStateLabelTests(_UpgBase):
    def test_missing_file_empty(self):
        self.assertEqual(U._prod_state_label(), "")

    def test_reads_state(self):
        self.write("hud_state.json", json.dumps({"state": "Speaking"}))
        self.assertEqual(U._prod_state_label(), "Speaking")

    def test_bad_json_empty(self):
        self.write("hud_state.json", "{not json")
        self.assertEqual(U._prod_state_label(), "")

    def test_non_dict_empty(self):
        self.write("hud_state.json", "[1,2,3]")
        self.assertEqual(U._prod_state_label(), "")


class WaitForProdIdleTests(_UpgBase):
    def test_idle_immediately(self):
        with mock.patch.object(U, "_prod_state_label", return_value="idle"):
            self.assertTrue(U._wait_for_prod_idle(timeout_s=1))

    def test_empty_counts_as_idle(self):
        with mock.patch.object(U, "_prod_state_label", return_value=""):
            self.assertTrue(U._wait_for_prod_idle(timeout_s=1))

    def test_times_out_when_busy(self):
        # always 'speaking' -> never idle. One loop iteration (hits the
        # time.sleep(poll_s) branch) then time.time advances past deadline.
        times = iter([1000.0, 1000.0, 1000.0, 9999.0, 9999.0])
        with mock.patch.object(U, "_prod_state_label", return_value="speaking"), \
             mock.patch.object(U.time, "time", lambda: next(times)):
            self.assertFalse(U._wait_for_prod_idle(timeout_s=0.05, poll_s=0.01))


class WaitForNewProdHeartbeatTests(_UpgBase):
    def test_heartbeat_seen(self):
        fake = _FakeBGM(self.tmp)
        fake.set_prod_running_sequence([True])
        fake._instances = {"a": {"role": "prod", "heartbeat_at": 12345.0}}
        with mock.patch.object(U, "_bgm", fake), \
             mock.patch.object(U.time, "time", return_value=12350.0):
            self.assertTrue(U._wait_for_new_prod_heartbeat(timeout_s=1))

    def test_no_bgm_returns_false(self):
        times = iter([0.0, 0.1, 100.0])
        with mock.patch.object(U, "_bgm", None), \
             mock.patch.object(U.time, "time", lambda: next(times)):
            self.assertFalse(U._wait_for_new_prod_heartbeat(timeout_s=0.01,
                                                            poll_s=0.001))

    def test_list_instances_exception_then_timeout(self):
        # prod_is_running True but list_instances raises -> insts={} -> no
        # heartbeat found -> loop then times out.
        fake = _FakeBGM(self.tmp)
        fake.set_prod_running_sequence([True, True])
        fake.list_instances = mock.Mock(side_effect=RuntimeError("io"))
        times = iter([1000.0, 1000.0, 1000.1, 9999.0])
        with mock.patch.object(U, "_bgm", fake), \
             mock.patch.object(U.time, "time", lambda: next(times)):
            self.assertFalse(U._wait_for_new_prod_heartbeat(timeout_s=0.05,
                                                            poll_s=0.01))

    def test_non_prod_and_nondict_entries_skipped(self):
        fake = _FakeBGM(self.tmp)
        fake.set_prod_running_sequence([True, True])
        fake._instances = {"a": "notadict",
                           "b": {"role": "staging", "heartbeat_at": 9999.0}}
        times = iter([1000.0, 1000.0, 1000.1, 9999.0])
        with mock.patch.object(U, "_bgm", fake), \
             mock.patch.object(U.time, "time", lambda: next(times)):
            self.assertFalse(U._wait_for_new_prod_heartbeat(timeout_s=0.05,
                                                            poll_s=0.01))

    def test_stale_heartbeat_times_out(self):
        fake = _FakeBGM(self.tmp)
        fake.set_prod_running_sequence([True, True])
        fake._instances = {"a": {"role": "prod", "heartbeat_at": 1.0}}
        times = iter([1000.0, 1000.0, 1000.1, 9999.0])
        with mock.patch.object(U, "_bgm", fake), \
             mock.patch.object(U.time, "time", lambda: next(times)):
            self.assertFalse(U._wait_for_new_prod_heartbeat(timeout_s=0.05,
                                                            poll_s=0.01))


# ═══════════════════════ run_blue_green_handoff ═══════════════════════════


class RunBlueGreenHandoffTests(_UpgBase):
    def _patch_helpers(self, bgm, stg):
        return [
            mock.patch.object(U, "_bgm", bgm),
            mock.patch.object(U, "_stg", stg),
        ]

    def test_missing_modules_aborts(self):
        with mock.patch.object(U, "_bgm", None), \
             mock.patch.object(U, "_stg", None):
            out = U.run_blue_green_handoff()
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage_failed"], "import")

    def test_precompile_failure_rolls_back(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        stg.precompile_result = (False, ["bob.py: SyntaxError"])
        ctxs = self._patch_helpers(bgm, stg)
        with ctxs[0], ctxs[1]:
            out = U.run_blue_green_handoff()
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage_failed"], "py_compile")
        self.assertIn("rollback", bgm.names())
        self.assertIn("signal_upgrade_aborted", bgm.names())

    def test_staging_spawn_failure_rolls_back(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        ctxs = self._patch_helpers(bgm, stg)
        with ctxs[0], ctxs[1], \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=None):
            out = U.run_blue_green_handoff()
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage_failed"], "spawn")
        self.assertIn("rollback", bgm.names())

    def test_smoke_failure_tears_down_staging(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        stg.smoke_result = {"ok": False, "stage_failed": "reply", "details": {"x": 1}}
        proc = _FakePopen(["x"])
        ctxs = self._patch_helpers(bgm, stg)
        with ctxs[0], ctxs[1], \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc):
            out = U.run_blue_green_handoff()
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage_failed"], "reply")
        self.assertTrue(proc._terminated)
        self.assertIn("rollback", bgm.names())

    def test_happy_path_full_handoff(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        # prod_is_running: False at the grace-loop check, then heartbeat path
        bgm.set_prod_running_sequence([False])
        proc = _FakePopen(["x"])
        ctxs = self._patch_helpers(bgm, stg)
        with ctxs[0], ctxs[1], \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc), \
             mock.patch.object(U, "_wait_for_prod_idle", return_value=True), \
             mock.patch.object(U, "relaunch_jarvis") as relaunch, \
             mock.patch.object(U, "_wait_for_new_prod_heartbeat", return_value=True):
            out = U.run_blue_green_handoff()
        self.assertTrue(out["ok"])
        self.assertIsNone(out["stage_failed"])
        # the full choreography fired in order
        names = bgm.names()
        self.assertIn("seed_staging_data", names)
        self.assertIn("signal_handoff", names)
        self.assertIn("promote_staging", names)
        relaunch.assert_called_once()

    def test_prod_does_not_yield_aborts(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        # prod stays running for every check in the grace loop + final check
        bgm.set_prod_running_sequence([True] * 50)
        proc = _FakePopen(["x"])
        ctxs = self._patch_helpers(bgm, stg)
        # advance time so the grace deadline passes quickly
        tvals = iter([1000.0] + [1000.0, 1001.0, 1002.0] + [9999.0] * 50)
        with ctxs[0], ctxs[1], \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc), \
             mock.patch.object(U, "_wait_for_prod_idle", return_value=True), \
             mock.patch.object(U.time, "time", lambda: next(tvals)):
            out = U.run_blue_green_handoff()
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage_failed"], "handoff_timeout")
        self.assertIn("signal_handoff_failure", bgm.names())
        self.assertTrue(proc._terminated)

    def test_precompile_failure_signal_exception_swallowed(self):
        # signal_upgrade_aborted raising must not propagate.
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        stg.precompile_result = (False, ["x: SyntaxError"])
        bgm.signal_upgrade_aborted = mock.Mock(side_effect=RuntimeError("voice down"))
        with mock.patch.object(U, "_bgm", bgm), mock.patch.object(U, "_stg", stg):
            out = U.run_blue_green_handoff()
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage_failed"], "py_compile")

    def test_smoke_failure_signal_exception_swallowed(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        stg.smoke_result = {"ok": False, "stage_failed": "boot", "details": {}}
        bgm.signal_upgrade_aborted = mock.Mock(side_effect=RuntimeError("x"))
        proc = _FakePopen(["x"])
        # proc.terminate also raising is swallowed by the bare except
        proc.terminate = mock.Mock(side_effect=RuntimeError("term fail"))
        with mock.patch.object(U, "_bgm", bgm), mock.patch.object(U, "_stg", stg), \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc):
            out = U.run_blue_green_handoff()
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage_failed"], "boot")

    def test_handoff_timeout_signal_failure_exception_swallowed(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        bgm.set_prod_running_sequence([True] * 50)
        bgm.signal_handoff_failure = mock.Mock(side_effect=RuntimeError("x"))
        proc = _FakePopen(["x"])
        proc.terminate = mock.Mock(side_effect=RuntimeError("x"))
        proc.wait = mock.Mock(side_effect=RuntimeError("x"))
        tvals = iter([1000.0, 1000.0, 1001.0, 9999.0] + [9999.0] * 50)
        with mock.patch.object(U, "_bgm", bgm), mock.patch.object(U, "_stg", stg), \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc), \
             mock.patch.object(U, "_wait_for_prod_idle", return_value=True), \
             mock.patch.object(U.time, "time", lambda: next(tvals)):
            out = U.run_blue_green_handoff()
        self.assertEqual(out["stage_failed"], "handoff_timeout")

    def test_spawn_failure_signal_exception_swallowed(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        bgm.signal_upgrade_aborted = mock.Mock(side_effect=RuntimeError("x"))
        with mock.patch.object(U, "_bgm", bgm), mock.patch.object(U, "_stg", stg), \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=None):
            out = U.run_blue_green_handoff()
        self.assertEqual(out["stage_failed"], "spawn")

    def test_happy_path_terminate_exceptions_swallowed(self):
        # on the success path the staging proc.terminate()/wait() raising is
        # swallowed by the bare excepts before promotion.
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        bgm.set_prod_running_sequence([False])
        proc = _FakePopen(["x"])
        proc.terminate = mock.Mock(side_effect=RuntimeError("x"))
        proc.wait = mock.Mock(side_effect=RuntimeError("x"))
        with mock.patch.object(U, "_bgm", bgm), mock.patch.object(U, "_stg", stg), \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc), \
             mock.patch.object(U, "_wait_for_prod_idle", return_value=True), \
             mock.patch.object(U, "relaunch_jarvis"), \
             mock.patch.object(U, "_wait_for_new_prod_heartbeat", return_value=True):
            out = U.run_blue_green_handoff()
        self.assertTrue(out["ok"])

    def test_stale_staging_lock_removed(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        bgm.set_prod_running_sequence([False])
        # pre-create a stale staging lock; the ceremony should remove it
        with open(bgm.STAGING_LOCK_FILE, "w") as f:
            f.write("stale")
        proc = _FakePopen(["x"])
        with mock.patch.object(U, "_bgm", bgm), mock.patch.object(U, "_stg", stg), \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc), \
             mock.patch.object(U, "_wait_for_prod_idle", return_value=True), \
             mock.patch.object(U, "relaunch_jarvis"), \
             mock.patch.object(U, "_wait_for_new_prod_heartbeat", return_value=True):
            out = U.run_blue_green_handoff()
        self.assertTrue(out["ok"])
        self.assertFalse(os.path.exists(bgm.STAGING_LOCK_FILE))

    def test_lock_remove_oserror_swallowed(self):
        # both the early staging-lock drop and the final lock cleanup hit
        # os.remove; if it raises OSError the ceremony must continue.
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        bgm.set_prod_running_sequence([False])
        for lk in (bgm.STAGING_LOCK_FILE, bgm.PROD_LOCK_FILE):
            with open(lk, "w") as f:
                f.write("stale")
        proc = _FakePopen(["x"])
        with mock.patch.object(U, "_bgm", bgm), mock.patch.object(U, "_stg", stg), \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc), \
             mock.patch.object(U, "_wait_for_prod_idle", return_value=True), \
             mock.patch.object(U, "relaunch_jarvis"), \
             mock.patch.object(U, "_wait_for_new_prod_heartbeat", return_value=True), \
             mock.patch.object(U.os, "remove", side_effect=OSError("locked")):
            out = U.run_blue_green_handoff()
        self.assertTrue(out["ok"])

    def test_happy_path_no_heartbeat_still_ok(self):
        bgm, stg = _FakeBGM(self.tmp), _FakeSTG()
        bgm.set_prod_running_sequence([False])
        proc = _FakePopen(["x"])
        ctxs = self._patch_helpers(bgm, stg)
        with ctxs[0], ctxs[1], \
             mock.patch.object(U, "spawn_staging_jarvis", return_value=proc), \
             mock.patch.object(U, "_wait_for_prod_idle", return_value=True), \
             mock.patch.object(U, "relaunch_jarvis"), \
             mock.patch.object(U, "_wait_for_new_prod_heartbeat", return_value=False):
            out = U.run_blue_green_handoff()
        self.assertTrue(out["ok"])
        self.assertFalse(out["details"]["new_prod_heartbeat"])


# ═══════════════════════ _append_changelog_entry ═════════════════════════


class AppendChangelogEntryTests(_UpgBase):
    def test_no_completed_tasks_noop(self):
        # tasks_before are all still open -> nothing completed -> no file
        self.write("jarvis_todo.md", "- [ ] still open\n")
        U._append_changelog_entry(["still open"], True, 0, 0, 0)
        self.assertFalse(self.exists("CHANGELOG.md"))

    def test_creates_changelog_and_bumps_version(self):
        # task completed: was unticked before, now ticked
        self.write("jarvis_todo.md", "- [x] add cool wish feature\n")
        self.write(os.path.join("data", "version.json"),
                   json.dumps({"version": "1.2.3"}))
        U._append_changelog_entry(["add cool wish feature"], True, 0, 0, 0)
        cl = self.read("CHANGELOG.md")
        # wish => minor bump
        self.assertIn("v1.3.0", cl)
        self.assertIn("Wish-list features", cl)
        vdata = json.loads(self.read(os.path.join("data", "version.json")))
        self.assertEqual(vdata["version"], "1.3.0")
        self.assertIn("last_upgrade_at", vdata)

    def test_patch_bump_for_bugfix_only(self):
        self.write("jarvis_todo.md", "- [x] [audit-v2] fix the bug\n")
        self.write(os.path.join("data", "version.json"),
                   json.dumps({"version": "2.0.0"}))
        U._append_changelog_entry(["[audit-v2] fix the bug"], True, 0, 0, 0)
        self.assertIn("v2.0.1", self.read("CHANGELOG.md"))

    def test_syntax_fail_warning_in_entry(self):
        self.write("jarvis_todo.md", "- [x] do a thing\n")
        U._append_changelog_entry(["do a thing"], False, 0, 0, 0)
        cl = self.read("CHANGELOG.md")
        self.assertIn("Syntax check FAILED", cl)

    def test_audit_findings_noted(self):
        self.write("jarvis_todo.md", "- [x] do a thing\n")
        U._append_changelog_entry(["do a thing"], True, 2, 3, 0)
        cl = self.read("CHANGELOG.md")
        self.assertIn("2 P0 finding", cl)
        self.assertIn("3 P1 finding", cl)

    def test_prepends_under_existing_header(self):
        self.write("jarvis_todo.md", "- [x] task A\n")
        self.write("CHANGELOG.md",
                   "# JARVIS Changelog\n\nintro\n\n---\n\n## v1.0.0 — old\n\nold body\n")
        U._append_changelog_entry(["task A"], True, 0, 0, 0)
        cl = self.read("CHANGELOG.md")
        # header preserved, new entry above the old one
        self.assertTrue(cl.startswith("# JARVIS Changelog"))
        self.assertLess(cl.index("task A"), cl.index("v1.0.0"))

    def test_corrupt_version_file_defaults(self):
        # version.json present but unparseable -> read except swallowed,
        # falls back to default 1.0.0 -> patch bump 1.0.1
        self.write("jarvis_todo.md", "- [x] plain task\n")
        self.write(os.path.join("data", "version.json"), "{broken json")
        U._append_changelog_entry(["plain task"], True, 0, 0, 0)
        self.assertIn("v1.0.1", self.read("CHANGELOG.md"))

    def test_default_version_when_no_version_file(self):
        self.write("jarvis_todo.md", "- [x] plain task\n")
        U._append_changelog_entry(["plain task"], True, 0, 0, 0)
        # default 1.0.0 -> patch bump -> 1.0.1
        self.assertIn("v1.0.1", self.read("CHANGELOG.md"))

    def test_exception_is_swallowed(self):
        # makedirs blows up inside the body -> caught, no raise
        self.write("jarvis_todo.md", "- [x] t\n")
        with mock.patch.object(U.os, "makedirs", side_effect=OSError("ro")):
            U._append_changelog_entry(["t"], True, 0, 0, 0)  # must not raise

    def test_many_items_truncates_to_30(self):
        ticked = "".join(f"- [x] research-{i} item number {i}\n" for i in range(40))
        self.write("jarvis_todo.md", ticked)
        before = [f"research-{i} item number {i}" for i in range(40)]
        U._append_changelog_entry(before, True, 0, 0, 0)
        cl = self.read("CHANGELOG.md")
        self.assertIn("and 10 more", cl)


# ═══════════════════════════════ main() ═══════════════════════════════════


class MainEntryTests(_UpgBase):
    def _run_main(self, argv):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("sys.stdout", out):
            try:
                U.main()
            except SystemExit:
                pass
        return out.getvalue()

    def test_force_trigger_gate(self):
        with mock.patch.object(U, "_stability_gate",
                               return_value={"verdict": "PASS", "ok": True,
                                             "details": "fine"}) as gate:
            txt = self._run_main(["upgrade_jarvis.py", "--force-trigger-gate-now"])
        gate.assert_called_once()
        self.assertIn("verdict=PASS", txt)

    def test_blue_green_dispatch(self):
        with mock.patch.object(U, "run_blue_green_handoff",
                               return_value={"ok": True, "stage_failed": None}) as bg:
            txt = self._run_main(["upgrade_jarvis.py", "--blue-green"])
        bg.assert_called_once()
        self.assertIn("verdict=OK", txt)

    def test_no_tasks_exits_early(self):
        self.write("jarvis_todo.md", "- [x] all done\n")
        with mock.patch.object(U, "find_claude_cli") as fc:
            txt = self._run_main(["upgrade_jarvis.py"])
        self.assertIn("Nothing to upgrade", txt)
        fc.assert_not_called()

    def test_dry_audit_skips_backup_and_kill(self):
        self.write("jarvis_todo.md", "- [ ] task one\n")
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase") as backup, \
             mock.patch.object(U, "kill_running_jarvis") as kill, \
             mock.patch.object(U, "spawn_claude_dry_audit",
                               return_value=_FakePopen(["x"])) as audit:
            txt = self._run_main(["upgrade_jarvis.py", "--dry-audit"])
        backup.assert_not_called()
        kill.assert_not_called()
        audit.assert_called_once()
        self.assertIn("DRY AUDIT", txt)

    def test_dry_audit_spawn_failure(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "spawn_claude_dry_audit", return_value=None):
            txt = self._run_main(["upgrade_jarvis.py", "--dry-audit"])
        self.assertIn("Failed", txt)

    def test_autonomous_no_relaunch(self):
        self.write("jarvis_todo.md", "- [ ] task one\n")
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase",
                               return_value=os.path.join(self.tmp, "bk")), \
             mock.patch.object(U, "kill_running_jarvis", return_value=1), \
             mock.patch.object(U, "spawn_claude_code",
                               return_value=_FakePopen(["x"])) as spawn:
            txt = self._run_main(["upgrade_jarvis.py"])
        spawn.assert_called_once()
        self.assertIn("manually run", txt)

    def test_autonomous_spawn_failure_aborts(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase",
                               return_value=os.path.join(self.tmp, "bk")), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=None):
            txt = self._run_main(["upgrade_jarvis.py"])
        self.assertIn("Aborting", txt)

    def test_relaunch_success_path(self):
        self.write("jarvis_todo.md", "- [ ] task one\n")
        self.write("bobert_companion.py", "print('ok')\n")   # valid syntax
        proc = _FakePopen(["x"])
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase",
                               return_value=os.path.join(self.tmp, "bk")), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=0)), \
             mock.patch.object(U, "_append_changelog_entry"), \
             mock.patch.object(U, "_revert_to_snapshot",
                               return_value=(True, "ok")), \
             mock.patch.object(U, "relaunch_jarvis") as relaunch:
            txt = self._run_main(["upgrade_jarvis.py", "--relaunch"])
        relaunch.assert_called_once()
        self.assertIn("JARVIS relaunched", txt)
        # summary file written
        self.assertTrue(self.exists(".last_upgrade_summary.json"))

    def test_relaunch_syntax_fail_triggers_revert(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        self.write("bobert_companion.py", "def broken(\n")   # invalid syntax
        proc = _FakePopen(["x"])
        backup = os.path.join(self.tmp, "bk")
        os.makedirs(backup, exist_ok=True)
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase", return_value=backup), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc), \
             mock.patch.object(U, "_append_changelog_entry"), \
             mock.patch.object(U, "_revert_to_snapshot",
                               return_value=(True, "reverted")) as revert, \
             mock.patch.object(U, "relaunch_jarvis"):
            txt = self._run_main(["upgrade_jarvis.py", "--relaunch"])
        # real py_compile runs against the broken file -> syntax_errors -> revert
        revert.assert_called_once_with(backup)
        self.assertIn("Syntax check FAILED", txt)

    def test_fallback_no_claude_cli(self):
        self.write("jarvis_todo.md", "- [ ] task one\n")
        with mock.patch.object(U, "find_claude_cli", return_value=None), \
             mock.patch.object(U, "backup_codebase",
                               return_value=os.path.join(self.tmp, "bk")), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "Popen", _FakePopen), \
             mock.patch.object(U.os, "startfile", create=True) as startfile, \
             mock.patch("builtins.input", return_value=""):
            txt = self._run_main(["upgrade_jarvis.py"])
        self.assertIn("Falling back to manual handoff", txt)
        startfile.assert_called_once()

    def _run_relaunch_with_auditor(self, audit_rc, report=None):
        """Drive the --relaunch path with a present auditor script and a given
        auditor returncode/report. Returns captured stdout."""
        self.write("jarvis_todo.md", "- [ ] t\n")
        self.write("bobert_companion.py", "print('ok')\n")
        self.write(os.path.join("tools", "audit_codebase.py"), "# auditor\n")
        if report is not None:
            self.write("audit_report.json", report)
        proc = _FakePopen(["x"])
        backup = os.path.join(self.tmp, "bk")
        os.makedirs(backup, exist_ok=True)
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase", return_value=backup), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=audit_rc)), \
             mock.patch.object(U, "_append_changelog_entry"), \
             mock.patch.object(U, "relaunch_jarvis"):
            return self._run_main(["upgrade_jarvis.py", "--relaunch"])

    def test_audit_clean_rc0(self):
        txt = self._run_relaunch_with_auditor(0)
        self.assertIn("Audit: CLEAN", txt)

    def test_audit_warn_rc2(self):
        txt = self._run_relaunch_with_auditor(
            2, json.dumps({"summary": {"p0": 0, "p1": 4, "p2": 0}}))
        self.assertIn("Audit: WARN", txt)
        self.assertIn("4 P1", txt)

    def test_audit_info_rc3(self):
        txt = self._run_relaunch_with_auditor(
            3, json.dumps({"summary": {"p0": 0, "p1": 0, "p2": 7}}))
        self.assertIn("Audit: INFO", txt)

    def test_audit_skipped_other_rc(self):
        txt = self._run_relaunch_with_auditor(-5)
        self.assertIn("Audit: skipped", txt)

    def test_audit_bad_report_json_swallowed(self):
        # report exists but is invalid JSON -> parse except swallowed, still OK
        txt = self._run_relaunch_with_auditor(0, "{not valid json")
        self.assertIn("Audit: CLEAN", txt)

    def test_single_stage_mode_banner(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        with mock.patch.object(U, "find_claude_cli", return_value=None), \
             mock.patch.object(U, "backup_codebase",
                               return_value=os.path.join(self.tmp, "bk")), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "Popen", _FakePopen), \
             mock.patch.object(U.os, "startfile", create=True), \
             mock.patch("builtins.input", return_value=""):
            txt = self._run_main(["upgrade_jarvis.py", "--single-stage"])
        self.assertIn("SINGLE-STAGE mode", txt)

    def test_relaunch_runs_auditor_p0_warning(self):
        # tools/audit_codebase.py present + report with P0 findings exercises
        # the auditor block AND the post-upgrade P0-warning relaunch branch.
        self.write("jarvis_todo.md", "- [ ] t\n")
        self.write("bobert_companion.py", "print('ok')\n")
        self.write(os.path.join("tools", "audit_codebase.py"), "# auditor\n")
        self.write("audit_report.json",
                   json.dumps({"summary": {"p0": 2, "p1": 1, "p2": 0}}))
        proc = _FakePopen(["x"])
        backup = os.path.join(self.tmp, "bk")
        os.makedirs(backup, exist_ok=True)
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase", return_value=backup), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=1)), \
             mock.patch.object(U, "_append_changelog_entry"), \
             mock.patch.object(U, "relaunch_jarvis") as relaunch:
            txt = self._run_main(["upgrade_jarvis.py", "--relaunch"])
        self.assertIn("Audit: FAIL", txt)
        self.assertIn("2 P0 finding", txt)
        # syntax was valid, so it still relaunches despite the P0
        relaunch.assert_called_once()

    def test_relaunch_auditor_timeout(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        self.write("bobert_companion.py", "print('ok')\n")
        self.write(os.path.join("tools", "audit_codebase.py"), "# auditor\n")
        proc = _FakePopen(["x"])
        backup = os.path.join(self.tmp, "bk")
        os.makedirs(backup, exist_ok=True)
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase", return_value=backup), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc), \
             mock.patch.object(U.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("audit", 120)), \
             mock.patch.object(U, "_append_changelog_entry"), \
             mock.patch.object(U, "relaunch_jarvis"):
            txt = self._run_main(["upgrade_jarvis.py", "--relaunch"])
        self.assertIn("[audit] timed out", txt)

    def test_relaunch_summary_write_failure_swallowed(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        self.write("bobert_companion.py", "print('ok')\n")
        proc = _FakePopen(["x"])
        backup = os.path.join(self.tmp, "bk")
        os.makedirs(backup, exist_ok=True)
        real_open = open

        def open_blowup(path, *a, **k):
            if str(path) == U.UPGRADE_SUMMARY_FILE:
                raise OSError("disk full")
            return real_open(path, *a, **k)

        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase", return_value=backup), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc), \
             mock.patch.object(U.subprocess, "run",
                               return_value=_FakeCompleted(returncode=0)), \
             mock.patch.object(U, "_append_changelog_entry"), \
             mock.patch.object(U, "relaunch_jarvis"), \
             mock.patch("builtins.open", side_effect=open_blowup):
            txt = self._run_main(["upgrade_jarvis.py", "--relaunch"])
        self.assertIn("Couldn't write upgrade summary", txt)

    def test_relaunch_syntax_fail_revert_exception_swallowed(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        self.write("bobert_companion.py", "def broken(\n")
        proc = _FakePopen(["x"])
        backup = os.path.join(self.tmp, "bk")
        os.makedirs(backup, exist_ok=True)
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase", return_value=backup), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc), \
             mock.patch.object(U, "_append_changelog_entry"), \
             mock.patch.object(U, "_revert_to_snapshot",
                               side_effect=RuntimeError("revert blew up")), \
             mock.patch.object(U, "relaunch_jarvis"):
            txt = self._run_main(["upgrade_jarvis.py", "--relaunch"])
        self.assertIn("Revert FAILED", txt)

    def test_relaunch_keyboard_interrupt(self):
        self.write("jarvis_todo.md", "- [ ] t\n")
        self.write("bobert_companion.py", "print('ok')\n")
        proc = _FakePopen(["x"])
        # proc.wait raises KeyboardInterrupt -> the main() handler prints
        proc.wait = mock.Mock(side_effect=KeyboardInterrupt())
        backup = os.path.join(self.tmp, "bk")
        os.makedirs(backup, exist_ok=True)
        with mock.patch.object(U, "find_claude_cli", return_value=r"C:\claude.exe"), \
             mock.patch.object(U, "backup_codebase", return_value=backup), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U, "spawn_claude_code", return_value=proc):
            txt = self._run_main(["upgrade_jarvis.py", "--relaunch"])
        self.assertIn("Interrupted", txt)

    def test_blue_green_dispatch_failure_exits_nonzero(self):
        with mock.patch.object(U, "run_blue_green_handoff",
                               return_value={"ok": False,
                                             "stage_failed": "smoke"}):
            txt = self._run_main(["upgrade_jarvis.py", "--blue-green"])
        self.assertIn("verdict=FAIL", txt)

    def test_fallback_clipboard_failure_prints_payload(self):
        self.write("jarvis_todo.md", "- [ ] task one\n")
        with mock.patch.object(U, "find_claude_cli", return_value=None), \
             mock.patch.object(U, "backup_codebase",
                               return_value=os.path.join(self.tmp, "bk")), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "Popen", side_effect=OSError("noclip")), \
             mock.patch.object(U.os, "startfile", create=True), \
             mock.patch("builtins.input", return_value=""):
            txt = self._run_main(["upgrade_jarvis.py"])
        self.assertIn("Paste this manually", txt)

    def test_fallback_startfile_failure_and_eof(self):
        # editor-open failure prints the error; input() EOFError is swallowed.
        self.write("jarvis_todo.md", "- [ ] task one\n")
        with mock.patch.object(U, "find_claude_cli", return_value=None), \
             mock.patch.object(U, "backup_codebase",
                               return_value=os.path.join(self.tmp, "bk")), \
             mock.patch.object(U, "kill_running_jarvis", return_value=0), \
             mock.patch.object(U.subprocess, "Popen", _FakePopen), \
             mock.patch.object(U.os, "startfile", create=True,
                               side_effect=OSError("no editor")), \
             mock.patch("builtins.input", side_effect=EOFError()):
            txt = self._run_main(["upgrade_jarvis.py"])
        self.assertIn("Could not open editor", txt)


if __name__ == "__main__":
    unittest.main()
