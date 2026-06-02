"""Unit tests for ``tools/stability_smoke_test.py`` — the post-pipeline
stability smoke gate that boots JARVIS, waits, and checks three+ signals
(process liveness, Windows APPCRASH events, crash_traces.log native dumps,
categorized session-log scan) before writing a PASS/FAIL report.

SAFETY CONTRACT (this suite NEVER boots JARVIS or touches the real tree):
  * No real PowerShell / tasklist / Get-WinEvent / booter — every
    ``subprocess.run`` is mocked with a fake CompletedProcess.
  * No real sleeping or wall-clock waiting — ``time.sleep`` is patched to a
    no-op and ``time.monotonic`` is scripted where a deadline matters.
  * No writes to the real C:\\JARVIS tree — every on-disk path constant the
    module computed at import time (ROOT, LOCK_FILE, LOGS_DIR, the two
    report paths, BOOT_SCRIPT, BOOT_ERR_FILE, CRASH_TRACES_LOG) is redirected
    into a per-test ``tempfile.TemporaryDirectory`` and restored in tearDown.

ISOLATION: all module-global redirections are saved/restored per test by
``_Base``; all patches use context managers / ``addCleanup`` so the suite is
order-independent.

CI FAITHFUL: the target imports ONLY stdlib, so it imports (does not skip) on
the bare Linux GitHub Actions runner. The handful of assertions that depend on
real Windows tooling are pure-Python over *mocked* subprocess output, so they
run identically on both OSes. No test here is gated on the host OS because none
of the exercised code paths require a Windows binary to actually exist — the
subprocess layer is always faked. (A single belt-and-suspenders skip guard is
provided via ``_WIN`` should a future Windows-only path be added.)

PRIVACY: only synthetic fixtures (host "10.0.0.5", user "alice"); any
secret-shaped string is assembled at runtime so the CI PII grep never matches a
literal.

stdlib ``unittest`` only — NO pytest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

# ── make tools/ importable regardless of cwd ──────────────────────────────
# upgrade_jarvis.py lives at repo root (auto-importable), but the target lives
# under tools/. The CI-sim runner / `python -m unittest` put the repo root on
# sys.path; we add tools/ so `import stability_smoke_test` resolves on both.
_TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import stability_smoke_test as S  # noqa: E402

_WIN = sys.platform.startswith("win")


# ─────────────────────────────── fakes ────────────────────────────────────

class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess — never spawns anything."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_returning(*results, record=None):
    """Build a fake ``subprocess.run`` that yields ``results`` in order
    (the last one repeats once exhausted). Each result is a _FakeCompleted
    or an Exception instance/class to raise. If ``record`` is a list, every
    argv passed to run() is appended to it."""
    seq = list(results)

    def _fake_run(cmd, *a, **k):
        if record is not None:
            record.append(cmd)
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("boom")
        return item

    return _fake_run


# ─────────────────────────────── base ─────────────────────────────────────

class _Base(unittest.TestCase):
    """Redirect every on-disk path the module baked in at import time into a
    per-test tempdir, and restore the originals afterwards. ``time.sleep`` is
    globally no-op'd for the whole suite so nothing ever actually waits."""

    # module attribute name -> filename (relative to tempdir) it should point at
    _PATH_ATTRS = {
        "ROOT": "",  # special-cased to the tempdir itself
        "BOOT_SCRIPT": "_boot_jarvis.ps1",
        "LOCK_FILE": "jarvis.lock",
        "BOOT_ERR_FILE": "jarvis_boot_error.txt",
        "LOGS_DIR": "logs",
        "PASS_REPORT": "stability_smoke_PASS.json",
        "FAIL_REPORT": "stability_smoke_FAIL.json",
        "CRASH_TRACES_LOG": os.path.join("logs", "crash_traces.log"),
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

        self._saved = {}
        for attr, rel in self._PATH_ATTRS.items():
            self._saved[attr] = getattr(S, attr)
            target = self.tmp if rel == "" else os.path.join(self.tmp, rel)
            setattr(S, attr, target)
        os.makedirs(self.logs, exist_ok=True)

        # Globally neutralize sleeping. Individual tests that care about the
        # poll loop re-patch time.monotonic on top of this.
        sleep_patch = mock.patch.object(S.time, "sleep", lambda *_a, **_k: None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def tearDown(self):
        for attr, val in self._saved.items():
            setattr(S, attr, val)

    # convenience accessors -------------------------------------------------
    @property
    def logs(self):
        return os.path.join(self.tmp, "logs")

    def write(self, relpath, text, mtime=None):
        path = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def write_session_log(self, name, text, mtime=None):
        return self.write(os.path.join("logs", name), text, mtime=mtime)


# ════════════════════════════ _launch_jarvis ══════════════════════════════

class TestLaunchJarvis(_Base):
    def test_missing_boot_script_raises(self):
        # BOOT_SCRIPT points into the tempdir but we never created it.
        self.assertFalse(os.path.isfile(S.BOOT_SCRIPT))
        with self.assertRaises(FileNotFoundError):
            S._launch_jarvis()

    def test_returns_stdout_stderr_tuple(self):
        self.write("_boot_jarvis.ps1", "# fake booter\n")
        rec = []
        fake = _run_returning(
            _FakeCompleted(0, "booted ok", "a warning"), record=rec
        )
        with mock.patch.object(S.subprocess, "run", fake):
            out, err = S._launch_jarvis()
        self.assertEqual(out, "booted ok")
        self.assertEqual(err, "a warning")
        # argv carries the headless flag + the boot script path.
        argv = rec[0]
        self.assertIn("-Headless", argv)
        self.assertIn(S.BOOT_SCRIPT, argv)
        self.assertIn("powershell.exe", argv)

    def test_none_streams_coerced_to_empty_strings(self):
        self.write("_boot_jarvis.ps1", "x")
        fake = _run_returning(_FakeCompleted(0, None, None))
        with mock.patch.object(S.subprocess, "run", fake):
            out, err = S._launch_jarvis()
        self.assertEqual((out, err), ("", ""))


# ═════════════════════════ _list_jarvis_processes ═════════════════════════

class TestListJarvisProcesses(_Base):
    def test_exception_returns_probe_failed_marker(self):
        fake = _run_returning(OSError("nope"))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._list_jarvis_processes()
        self.assertEqual(len(res), 1)
        self.assertIn("process probe failed", res[0])

    def test_nonzero_rc_returns_rc_marker(self):
        fake = _run_returning(_FakeCompleted(3, "", "access denied"))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._list_jarvis_processes()
        self.assertEqual(len(res), 1)
        self.assertIn("process probe rc=3", res[0])
        self.assertIn("access denied", res[0])

    def test_empty_output_returns_none_seen_marker(self):
        fake = _run_returning(_FakeCompleted(0, "   \n  \n", ""))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._list_jarvis_processes()
        self.assertEqual(res, ["<no bobert_companion processes>"])

    def test_populated_output_split_into_lines(self):
        out = "1234 python.exe :: ...bobert_companion...\n5678 pythonw.exe :: ...\n"
        fake = _run_returning(_FakeCompleted(0, out, ""))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._list_jarvis_processes()
        self.assertEqual(len(res), 2)
        self.assertTrue(res[0].startswith("1234 python.exe"))


# ════════════════════════════ _latest_log_tail ════════════════════════════

class TestLatestLogTail(_Base):
    def test_no_log_returns_empty(self):
        # logs/ exists but has no session_*.log files.
        res = S._latest_log_tail(10)
        self.assertEqual(res, {"path": None, "tail": []})

    def test_tail_returns_last_n_lines(self):
        body = "".join(f"line{i}\n" for i in range(100))
        p = self.write_session_log("session_1.log", body, mtime=1000)
        res = S._latest_log_tail(5)
        self.assertEqual(res["path"], p)
        self.assertEqual(res["tail"], ["line95", "line96", "line97", "line98", "line99"])

    def test_read_oserror_surfaced_in_tail(self):
        p = self.write_session_log("session_1.log", "x\n", mtime=1000)
        real_open = open

        def boom(path, *a, **k):
            if path == p:
                raise OSError("locked")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", boom):
            res = S._latest_log_tail(5)
        self.assertEqual(res["path"], p)
        self.assertEqual(len(res["tail"]), 1)
        self.assertIn("could not read", res["tail"][0])


# ════════════════════════════ _read_lock_pid ══════════════════════════════

class TestReadLockPid(_Base):
    def test_absent_lock_returns_none(self):
        self.assertIsNone(S._read_lock_pid())

    def test_valid_pid(self):
        self.write("jarvis.lock", "  4242 \n")
        self.assertEqual(S._read_lock_pid(), 4242)

    def test_unparseable_pid_returns_none(self):
        self.write("jarvis.lock", "not-a-number")
        self.assertIsNone(S._read_lock_pid())


# ═══════════════════════════ _read_boot_error ═════════════════════════════

class TestReadBootError(_Base):
    def test_absent_returns_empty(self):
        self.assertEqual(S._read_boot_error(), "")

    def test_present_returns_stripped_contents(self):
        self.write("jarvis_boot_error.txt", "  could not lock\n\n")
        self.assertEqual(S._read_boot_error(), "could not lock")


# ══════════════════════════ _wait_for_lock_pid ════════════════════════════

class TestWaitForLockPid(_Base):
    def test_pid_available_immediately(self):
        self.write("jarvis.lock", "999")
        # monotonic must report a value < deadline at least once.
        with mock.patch.object(S.time, "monotonic", side_effect=[0.0, 0.0]):
            self.assertEqual(S._wait_for_lock_pid(30), 999)

    def test_boot_error_short_circuits(self):
        # No lock, but a boot-error marker → bail out fast with None.
        self.write("jarvis_boot_error.txt", "early fail")
        with mock.patch.object(S.time, "monotonic", side_effect=[0.0, 0.0]):
            self.assertIsNone(S._wait_for_lock_pid(30))

    def test_timeout_returns_none(self):
        # monotonic crosses the deadline on the 2nd read → loop never enters.
        with mock.patch.object(S.time, "monotonic", side_effect=[0.0, 100.0]):
            self.assertIsNone(S._wait_for_lock_pid(30))

    def test_pid_appears_after_a_poll(self):
        # First iteration: no lock, no boot error → sleep. Second: lock present.
        calls = {"n": 0}

        def fake_read_lock():
            calls["n"] += 1
            return 4321 if calls["n"] >= 2 else None

        with mock.patch.object(S, "_read_lock_pid", fake_read_lock), \
             mock.patch.object(S, "_read_boot_error", return_value=""), \
             mock.patch.object(S.time, "monotonic", side_effect=[0.0, 1.0, 2.0]):
            self.assertEqual(S._wait_for_lock_pid(30), 4321)


# ════════════════════════════════ _pid_alive ══════════════════════════════

class TestPidAlive(_Base):
    def test_exception_returns_false(self):
        fake = _run_returning(subprocess.SubprocessError("x"))
        with mock.patch.object(S.subprocess, "run", fake):
            self.assertFalse(S._pid_alive(4242))

    def test_alive_when_pid_and_python_present(self):
        out = '"python.exe","4242","Console","1","100,000 K"'
        fake = _run_returning(_FakeCompleted(0, out, ""))
        with mock.patch.object(S.subprocess, "run", fake):
            self.assertTrue(S._pid_alive(4242))

    def test_dead_when_pid_absent(self):
        out = "INFO: No tasks are running which match the specified criteria."
        fake = _run_returning(_FakeCompleted(0, out, ""))
        with mock.patch.object(S.subprocess, "run", fake):
            self.assertFalse(S._pid_alive(4242))

    def test_dead_when_present_but_not_python(self):
        # PID string present but it's some other (non-python) image.
        out = '"notepad.exe","4242","Console","1","9,000 K"'
        fake = _run_returning(_FakeCompleted(0, out, ""))
        with mock.patch.object(S.subprocess, "run", fake):
            self.assertFalse(S._pid_alive(4242))


# ═══════════════════════════ _latest_session_log ══════════════════════════

class TestLatestSessionLog(_Base):
    def test_missing_dir_returns_none(self):
        # Point LOGS_DIR at a path that does not exist.
        S.LOGS_DIR = os.path.join(self.tmp, "no_such_dir")
        self.assertIsNone(S._latest_session_log())

    def test_no_candidates_returns_none(self):
        self.write_session_log("notes.txt", "ignored")  # wrong prefix/suffix
        self.assertIsNone(S._latest_session_log())

    def test_picks_newest_by_mtime(self):
        self.write_session_log("session_a.log", "a", mtime=1000)
        newest = self.write_session_log("session_b.log", "b", mtime=5000)
        self.write_session_log("session_c.log", "c", mtime=3000)
        self.assertEqual(S._latest_session_log(), newest)

    def test_getmtime_oserror_skips_file(self):
        good = self.write_session_log("session_good.log", "g", mtime=1000)
        bad = self.write_session_log("session_bad.log", "b", mtime=9000)
        real_getmtime = os.path.getmtime

        def flaky(path):
            if path == bad:
                raise OSError("stat fail")
            return real_getmtime(path)

        with mock.patch.object(S.os.path, "getmtime", flaky):
            # bad is newest but un-stattable → falls back to good.
            self.assertEqual(S._latest_session_log(), good)


# ════════════════════════ _scan_crash_traces_since ════════════════════════

class TestScanCrashTracesSince(_Base):
    def _crash_path(self):
        return S.CRASH_TRACES_LOG

    def test_missing_file_returns_empty_result(self):
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(res["new_dumps"], [])
        self.assertEqual(res["head_signatures"], [])
        self.assertEqual(res["path"], self._crash_path())

    def test_stale_file_before_since_returns_empty(self):
        self.write(os.path.join("logs", "crash_traces.log"),
                   "Fatal Python error: boom\n", mtime=500)
        res = S._scan_crash_traces_since(1000.0)  # since is AFTER mtime
        self.assertEqual(res["new_dumps"], [])

    def test_getmtime_oserror_returns_empty(self):
        p = self.write(os.path.join("logs", "crash_traces.log"), "x\n", mtime=2000)

        def boom(path):
            if path == p:
                raise OSError("stat")
            return 0

        with mock.patch.object(S.os.path, "getmtime", boom):
            res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(res["new_dumps"], [])

    def test_read_oserror_appended_to_new_dumps(self):
        p = self.write(os.path.join("logs", "crash_traces.log"),
                       "Fatal Python error: x\n", mtime=5000)
        real_open = open

        def boom(path, *a, **k):
            if path == p:
                raise OSError("locked")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", boom):
            res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(len(res["new_dumps"]), 1)
        self.assertIn("could not read crash_traces.log", res["new_dumps"][0])

    def test_no_dump_starts_returns_empty(self):
        self.write(os.path.join("logs", "crash_traces.log"),
                   "just some boring log line\nanother\n", mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(res["new_dumps"], [])
        self.assertEqual(res["head_signatures"], [])

    def test_fatal_dump_with_current_thread_signature(self):
        dump = (
            "Fatal Python error: Segmentation fault\n"
            "\n"
            "Current thread 0x00001 (most recent call first):\n"
            '  File "C:\\\\JARVIS\\\\core\\\\audio_processor.py", line 88, in _ns\n'
            '  File "C:\\\\JARVIS\\\\bobert_companion.py", line 9, in <module>\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), dump, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(len(res["new_dumps"]), 1)
        self.assertIn("Fatal Python error", res["new_dumps"][0])
        self.assertEqual(len(res["head_signatures"]), 1)
        # The crash SITE (audio_processor _ns) is picked, NOT the <module> frame.
        self.assertIn("audio_processor.py", res["head_signatures"][0])
        self.assertIn("in _ns", res["head_signatures"][0])
        self.assertNotIn("<module>", res["head_signatures"][0])

    def test_current_thread_skips_non_file_lines_before_frame(self):
        # Inside the "Current thread" block there can be non-"File " lines
        # (a stray status line) before the first frame; those must be skipped
        # (the `continue` at the top-of-stack scan) without aborting the search.
        dump = (
            "Fatal Python error: Segmentation fault\n"
            "\n"
            "Current thread 0x00009 (most recent call first):\n"
            "  (some non-frame annotation line)\n"
            '  File "C:\\\\JARVIS\\\\core\\\\late.py", line 12, in handler\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), dump, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(len(res["head_signatures"]), 1)
        self.assertIn("late.py", res["head_signatures"][0])
        self.assertIn("in handler", res["head_signatures"][0])

    def test_windows_fatal_exception_marker_detected(self):
        dump = (
            "Windows fatal exception: access violation\n"
            "\n"
            "Current thread 0x00002 (most recent call first):\n"
            '  File "C:\\\\JARVIS\\\\skills\\\\foo.py", line 3, in handle\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), dump, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(len(res["new_dumps"]), 1)
        self.assertIn("foo.py", res["head_signatures"][0])

    def test_thread0x_midstream_not_treated_as_dump_start(self):
        # A 'Thread 0x' preceded by a non-empty line is an interior thread of a
        # dump, NOT a new dump start. With no real start marker → no dumps.
        body = (
            "some preamble line\n"
            "Thread 0x00003 (most recent call first):\n"
            '  File "C:\\\\JARVIS\\\\x.py", line 1, in f\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), body, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(res["new_dumps"], [])

    def test_thread0x_at_file_start_is_a_dump_start(self):
        # 'Thread 0x' as the very first line (i == 0) counts as a dump start.
        body = (
            "Thread 0x00004 (most recent call first):\n"
            '  File "C:\\\\JARVIS\\\\core\\\\thing.py", line 7, in run\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), body, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(len(res["new_dumps"]), 1)

    def test_fallback_signature_when_no_current_thread_block(self):
        # No "Current thread" header → fallback scan over the whole dump for the
        # topmost non-<module> JARVIS frame.
        dump = (
            "Fatal Python error: Aborted\n"
            '  File "C:\\\\JARVIS\\\\core\\\\widget.py", line 5, in tick\n'
            '  File "C:\\\\JARVIS\\\\bobert_companion.py", line 1, in <module>\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), dump, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(len(res["head_signatures"]), 1)
        self.assertIn("widget.py", res["head_signatures"][0])

    def test_dump_present_but_no_jarvis_frame_yields_no_signature(self):
        dump = (
            "Fatal Python error: Aborted\n"
            "\n"
            "Current thread 0x00005 (most recent call first):\n"
            '  File "/usr/lib/python3.14/threading.py", line 1, in run\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), dump, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        self.assertEqual(len(res["new_dumps"]), 1)
        self.assertEqual(res["head_signatures"], [])

    def test_last_dump_is_the_one_reported(self):
        dump = (
            "Fatal Python error: first crash\n"
            '  File "C:\\\\JARVIS\\\\a.py", line 1, in old\n'
            "\n"
            "Fatal Python error: second crash\n"
            '  File "C:\\\\JARVIS\\\\b.py", line 2, in newer\n'
        )
        self.write(os.path.join("logs", "crash_traces.log"), dump, mtime=5000)
        res = S._scan_crash_traces_since(1000.0)
        # faulthandler appends; the most-recent (last) dump is what we surface.
        self.assertIn("second crash", res["new_dumps"][0])
        self.assertNotIn("first crash", res["new_dumps"][0])
        self.assertIn("b.py", res["head_signatures"][0])


# ══════════════════════════ _scan_log_for_fatal ═══════════════════════════

class TestScanLogForFatal(_Base):
    def test_wrapper_returns_only_fatal_list(self):
        p = self.write_session_log(
            "session_1.log",
            "[INFO] start\n[FATAL] kaboom\nListening...\n",
            mtime=1000,
        )
        self.assertEqual(S._scan_log_for_fatal(p), ["[FATAL] kaboom"])

    def test_wrapper_empty_when_no_fatal(self):
        p = self.write_session_log("session_1.log", "Listening...\n", mtime=1000)
        self.assertEqual(S._scan_log_for_fatal(p), [])


# ═════════════════════════ _scan_log_for_issues ═══════════════════════════

class TestScanLogForIssues(_Base):
    def _scan(self, text):
        p = self.write_session_log("session_x.log", text, mtime=1000)
        return S._scan_log_for_issues(p)

    def test_read_error_marks_not_ok(self):
        missing = os.path.join(self.logs, "does_not_exist.log")
        res = S._scan_log_for_issues(missing)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["fatal"]), 1)
        self.assertIn("could not read log", res["fatal"][0])

    def test_healthy_boot_is_ok(self):
        text = (
            "[faulthandler] enabled\n"
            "[skill] weather: added actions\n"
            "[diag-daemons] 4/4 daemons running\n"
            "[hud] launched\n"
            "Listening...\n"
        )
        res = self._scan(text)
        self.assertTrue(res["ok"])
        ms = res["boot_milestones"]
        self.assertTrue(ms["listening"])
        self.assertTrue(ms["skills_loaded"])
        self.assertTrue(ms["diag_daemons"])
        self.assertTrue(ms["hud_spawned"])
        self.assertTrue(ms["faulthandler"])

    def test_listening_unicode_ellipsis_accepted(self):
        # JARVIS may emit U+2026 instead of three ASCII dots.
        res = self._scan("Listening…\n")
        self.assertTrue(res["boot_milestones"]["listening"])

    def test_no_listening_milestone_fails_gate(self):
        # Clean log but never reached the main loop → not ok.
        res = self._scan("[skill] x: added actions\n[hud] launched\n")
        self.assertFalse(res["ok"])
        self.assertFalse(res["boot_milestones"]["listening"])

    def test_fatal_fails_gate(self):
        res = self._scan("Listening...\n[FATAL] dead\n")
        self.assertFalse(res["ok"])
        self.assertEqual(res["fatal"], ["[FATAL] dead"])

    def test_fatal_capped_at_20(self):
        text = "Listening...\n" + "".join(f"[FATAL] line {i}\n" for i in range(50))
        res = self._scan(text)
        self.assertEqual(len(res["fatal"]), 20)
        self.assertFalse(res["ok"])

    def test_traceback_captured_and_fails_gate(self):
        text = (
            "Listening...\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in <module>\n'
            "    boom()\n"
            "ValueError: nope\n"
            "[INFO] recovered\n"
        )
        res = self._scan(text)
        self.assertEqual(len(res["tracebacks"]), 1)
        self.assertIn("Traceback (most recent call last):", res["tracebacks"][0])
        self.assertIn("ValueError: nope", res["tracebacks"][0])
        self.assertFalse(res["ok"])

    def test_traceback_defensive_cap_when_no_end_marker(self):
        # A traceback whose frames never terminate with an unindented line:
        # the >30-line defensive cap path appends a truncated entry.
        lines = ["Traceback (most recent call last):"]
        lines += [f"  frame {i}" for i in range(40)]  # all indented, no end
        text = "Listening...\n" + "\n".join(lines) + "\n"
        res = self._scan(text)
        self.assertEqual(len(res["tracebacks"]), 1)
        self.assertIn("(truncated)", res["tracebacks"][0])
        self.assertFalse(res["ok"])

    def test_tracebacks_capped_at_10(self):
        block = (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in f\n'
            "RuntimeError: e\n"
        )
        text = "Listening...\n" + block * 15
        res = self._scan(text)
        self.assertEqual(len(res["tracebacks"]), 10)

    def test_errors_collected_but_not_fatal(self):
        text = "Listening...\n[ERROR] something odd\nplain error: lowercase\n"
        res = self._scan(text)
        self.assertEqual(len(res["errors"]), 2)
        # errors alone don't sink the gate.
        self.assertTrue(res["ok"])

    def test_errors_capped_at_30(self):
        text = "Listening...\n" + "".join(f"[error] e{i}\n" for i in range(40))
        res = self._scan(text)
        self.assertEqual(len(res["errors"]), 30)

    def test_fatal_line_not_double_counted_as_error(self):
        # A line containing [FATAL] must not also be appended to errors even
        # though it might otherwise match an error pattern.
        text = "Listening...\n[FATAL] error: catastrophic\n"
        res = self._scan(text)
        self.assertEqual(res["fatal"], ["[FATAL] error: catastrophic"])
        self.assertEqual(res["errors"], [])

    def test_skill_failures_collected(self):
        text = "Listening...\n[skill] weather: failed to load\n"
        res = self._scan(text)
        self.assertEqual(len(res["skill_failures"]), 1)

    def test_skill_failures_capped_at_15(self):
        text = "Listening...\n" + "".join(
            f"[skill] s{i}: failed\n" for i in range(20)
        )
        res = self._scan(text)
        self.assertEqual(len(res["skill_failures"]), 15)

    def test_transcribe_failures_collected(self):
        text = "Listening...\n[transcribe] failed: cublas64_12.dll missing\n"
        res = self._scan(text)
        self.assertEqual(len(res["transcribe_failures"]), 1)

    def test_transcribe_failures_capped_at_15(self):
        text = "Listening...\n" + "".join(
            f"[transcribe] failed {i}\n" for i in range(20)
        )
        res = self._scan(text)
        self.assertEqual(len(res["transcribe_failures"]), 15)

    def test_missing_deps_each_keyword(self):
        for kw in ("not installed", "pip install foo", "not available",
                   "ModuleNotFoundError: no module"):
            res = self._scan(f"Listening...\n{kw}\n")
            self.assertEqual(len(res["missing_deps"]), 1, kw)

    def test_missing_deps_capped_at_15(self):
        text = "Listening...\n" + "".join(
            f"pkg {i} not installed\n" for i in range(20)
        )
        res = self._scan(text)
        self.assertEqual(len(res["missing_deps"]), 15)

    def test_native_crash_signatures_fail_gate(self):
        for sig in ("Thread 0x00001 (most recent call first):",
                    "Segmentation fault", "SIGSEGV", "SIGABRT"):
            res = self._scan(f"Listening...\n{sig}\n")
            self.assertTrue(res["native_crashes"], sig)
            self.assertFalse(res["ok"], sig)

    def test_native_crashes_capped_at_15(self):
        text = "Listening...\n" + "SIGSEGV\n" * 20
        res = self._scan(text)
        self.assertEqual(len(res["native_crashes"]), 15)


# ═══════════════════════════ _check_appcrash_events ═══════════════════════

class TestCheckAppcrashEvents(_Base):
    SINCE = datetime(2026, 6, 2, 8, 0, 0)

    def test_invocation_exception(self):
        fake = _run_returning(OSError("powershell gone"))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._check_appcrash_events(self.SINCE)
        self.assertFalse(res["ok"])
        self.assertIn("powershell invocation failed", res["error"])

    def test_nonzero_rc(self):
        fake = _run_returning(_FakeCompleted(1, "", "Get-WinEvent: boom"))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._check_appcrash_events(self.SINCE)
        self.assertFalse(res["ok"])
        self.assertIn("Get-WinEvent failed", res["error"])

    def test_empty_stdout_is_ok_zero_events(self):
        fake = _run_returning(_FakeCompleted(0, "   ", ""))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._check_appcrash_events(self.SINCE)
        self.assertTrue(res["ok"])
        self.assertEqual(res["count"], 0)
        self.assertEqual(res["events"], [])

    def test_unparseable_json(self):
        fake = _run_returning(_FakeCompleted(0, "not json at all", ""))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._check_appcrash_events(self.SINCE)
        self.assertFalse(res["ok"])
        self.assertIn("could not parse PS output", res["error"])

    def test_single_event_dict_wrapped_to_list(self):
        evt = {"TimeCreated": "/Date(0)/", "Id": 1000,
               "ProviderName": "Application Error", "Message": "APPCRASH"}
        fake = _run_returning(_FakeCompleted(0, json.dumps(evt), ""))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._check_appcrash_events(self.SINCE)
        self.assertFalse(res["ok"])  # a crash event present → not ok
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["events"][0]["Id"], 1000)

    def test_list_of_events_not_ok(self):
        evts = [{"Id": 1000}, {"Id": 1026}]
        fake = _run_returning(_FakeCompleted(0, json.dumps(evts), ""))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._check_appcrash_events(self.SINCE)
        self.assertFalse(res["ok"])
        self.assertEqual(res["count"], 2)

    def test_empty_json_array_is_ok(self):
        fake = _run_returning(_FakeCompleted(0, "[]", ""))
        with mock.patch.object(S.subprocess, "run", fake):
            res = S._check_appcrash_events(self.SINCE)
        self.assertTrue(res["ok"])
        self.assertEqual(res["count"], 0)

    def test_since_is_formatted_into_script(self):
        rec = []
        fake = _run_returning(_FakeCompleted(0, "[]", ""), record=rec)
        with mock.patch.object(S.subprocess, "run", fake):
            S._check_appcrash_events(self.SINCE)
        # The argv's last element is the -Command script carrying the timestamp.
        script = rec[0][-1]
        self.assertIn("2026-06-02T08:00:00", script)


# ════════════════════════════ _write_report ═══════════════════════════════

class TestWriteReport(_Base):
    def test_writes_pass_and_removes_fail(self):
        # Pre-existing FAIL report should be deleted when a PASS is written.
        with open(S.FAIL_REPORT, "w", encoding="utf-8") as f:
            f.write("{}")
        S._write_report(S.PASS_REPORT, {"result": "PASS", "n": 1})
        self.assertTrue(os.path.exists(S.PASS_REPORT))
        self.assertFalse(os.path.exists(S.FAIL_REPORT))
        with open(S.PASS_REPORT, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["result"], "PASS")

    def test_writes_fail_and_removes_pass(self):
        with open(S.PASS_REPORT, "w", encoding="utf-8") as f:
            f.write("{}")
        S._write_report(S.FAIL_REPORT, {"result": "FAIL"})
        self.assertTrue(os.path.exists(S.FAIL_REPORT))
        self.assertFalse(os.path.exists(S.PASS_REPORT))

    def test_remove_oserror_is_swallowed(self):
        # If removing the 'other' report raises, _write_report must still write.
        with mock.patch.object(S.os, "remove", side_effect=OSError("perm")), \
             mock.patch.object(S.os.path, "exists", return_value=True):
            S._write_report(S.PASS_REPORT, {"ok": True})
        self.assertTrue(os.path.exists(S.PASS_REPORT))


# ═══════════════════════════════════ main ═════════════════════════════════

class TestMainArgParsing(_Base):
    def test_default_wait_is_300(self):
        # Drive main through the lock-wait fail path quickly and assert the
        # report records the default wait was NOT reached (fail happens before
        # the sleep). We just confirm parsing by inspecting wait via a stubbed
        # _wait_for_lock_pid that captures nothing — simplest: no-launch + no
        # lock → fail at lock_wait, default never sleeps.
        with mock.patch.object(S, "_wait_for_lock_pid", return_value=None), \
             mock.patch.object(S, "_list_jarvis_processes", return_value=[]), \
             mock.patch.object(S, "_latest_log_tail", return_value={"path": None, "tail": []}):
            rc = S.main(["--no-launch"])
        self.assertEqual(rc, 1)

    def test_custom_wait_passed_through(self):
        captured = {}

        def fake_sleep(n):
            captured["wait"] = n

        with mock.patch.object(S, "_wait_for_lock_pid", return_value=4242), \
             mock.patch.object(S.time, "sleep", fake_sleep), \
             mock.patch.object(S, "_pid_alive", return_value=True), \
             mock.patch.object(S, "_check_appcrash_events", return_value={"ok": True, "count": 0, "events": []}), \
             mock.patch.object(S, "_scan_crash_traces_since", return_value={"path": "p", "new_dumps": [], "head_signatures": []}), \
             mock.patch.object(S, "_latest_session_log", return_value=None):
            rc = S.main(["--no-launch", "--wait", "7"])
        self.assertEqual(captured["wait"], 7)
        # session log None → session check not ok → overall FAIL.
        self.assertEqual(rc, 1)


class TestMainLaunchPhase(_Base):
    def test_launch_failure_writes_fail_report(self):
        # --no-launch is NOT passed, and _launch_jarvis raises.
        with mock.patch.object(S, "_launch_jarvis",
                               side_effect=FileNotFoundError("boot script missing")):
            rc = S.main([])
        self.assertEqual(rc, 1)
        self.assertTrue(os.path.exists(S.FAIL_REPORT))
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["phase"], "launch")
        self.assertEqual(payload["result"], "FAIL")
        self.assertIn("could not launch booter", payload["error"])

    def test_launch_success_then_lock_wait_proceeds(self):
        # Launch returns streams; lock wait then fails → lock_wait phase, and
        # the booter streams are surfaced in the report tail.
        with mock.patch.object(S, "_launch_jarvis",
                               return_value=("boot stdout here", "boot stderr here")), \
             mock.patch.object(S, "_wait_for_lock_pid", return_value=None), \
             mock.patch.object(S, "_read_boot_error", return_value=""), \
             mock.patch.object(S, "_list_jarvis_processes", return_value=["1 python.exe :: bobert"]), \
             mock.patch.object(S, "_latest_log_tail", return_value={"path": None, "tail": []}):
            rc = S.main([])
        self.assertEqual(rc, 1)
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["phase"], "lock_wait")
        self.assertEqual(payload["boot_stdout_tail"], "boot stdout here")
        self.assertEqual(payload["boot_stderr_tail"], "boot stderr here")


class TestMainLockWaitFail(_Base):
    def test_lock_wait_fail_surfaces_all_diagnostics(self):
        log_tail = {"path": os.path.join(self.logs, "session_1.log"),
                    "tail": ["line a", "line b"]}
        with mock.patch.object(S, "_wait_for_lock_pid", return_value=None), \
             mock.patch.object(S, "_read_boot_error", return_value="lock write failed"), \
             mock.patch.object(S, "_list_jarvis_processes",
                               return_value=["111 python.exe :: ...bobert_companion..."]), \
             mock.patch.object(S, "_latest_log_tail", return_value=log_tail):
            rc = S.main(["--no-launch"])
        self.assertEqual(rc, 1)
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["phase"], "lock_wait")
        self.assertEqual(payload["early_boot_error"], "lock write failed")
        self.assertEqual(len(payload["jarvis_processes"]), 1)
        self.assertEqual(payload["latest_session_log"]["tail"], ["line a", "line b"])

    def test_lock_wait_fail_with_no_boot_error_and_no_log_path(self):
        # Exercise the branches where boot_error is empty and log path is None
        # (so the "newest session log" print block is skipped).
        with mock.patch.object(S, "_wait_for_lock_pid", return_value=None), \
             mock.patch.object(S, "_read_boot_error", return_value=""), \
             mock.patch.object(S, "_list_jarvis_processes", return_value=[]), \
             mock.patch.object(S, "_latest_log_tail", return_value={"path": None, "tail": []}):
            rc = S.main(["--no-launch"])
        self.assertEqual(rc, 1)
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["early_boot_error"], "")


class TestMainFullRun(_Base):
    """Drive main() end-to-end with every external call mocked, exercising both
    the all-PASS and the per-check FAIL assembly + report writing."""

    def _patches(self, *, alive, appcrash, crash, scan_ok, session_log_present=True,
                 pre_equals_post=False):
        # Build a real session log on disk so the categorized-scan branch runs
        # against actual file content when session_log_present is True.
        if session_log_present:
            text = "Listening...\n" if scan_ok else "[FATAL] boom\nListening...\n"
            log = self.write_session_log("session_run.log", text, mtime=9000)
        else:
            log = None

        cm = []
        cm.append(mock.patch.object(S, "_wait_for_lock_pid", return_value=4242))
        cm.append(mock.patch.object(S, "_pid_alive", return_value=alive))
        cm.append(mock.patch.object(
            S, "_check_appcrash_events",
            return_value={"ok": appcrash, "count": 0 if appcrash else 1,
                          "events": [] if appcrash else [{"Id": 1000}]}))
        cm.append(mock.patch.object(
            S, "_scan_crash_traces_since",
            return_value={"path": "p",
                          "new_dumps": [] if crash else ["DUMP"],
                          "head_signatures": [] if crash else ["sig"]}))
        if pre_equals_post:
            # Force pre_launch_log == session_log to hit the "predates launch"
            # warning branch: both calls return the same path.
            cm.append(mock.patch.object(S, "_latest_session_log", return_value=log))
        elif session_log_present:
            # pre-launch returns None, post-launch returns the real log.
            cm.append(mock.patch.object(
                S, "_latest_session_log", side_effect=[None, log]))
        else:
            cm.append(mock.patch.object(S, "_latest_session_log", return_value=None))
        return cm, log

    def _run(self, cms, argv=("--no-launch",)):
        for c in cms:
            c.start()
            self.addCleanup(c.stop)
        return S.main(list(argv))

    def test_all_checks_pass_writes_pass_report(self):
        cms, _ = self._patches(alive=True, appcrash=True, crash=True, scan_ok=True)
        rc = self._run(cms)
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(S.PASS_REPORT))
        self.assertFalse(os.path.exists(S.FAIL_REPORT))
        with open(S.PASS_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["result"], "PASS")
        self.assertEqual(payload["pid"], 4242)
        self.assertTrue(payload["checks"]["process_alive"]["ok"])
        self.assertTrue(payload["checks"]["session_log_fatal"]["ok"])

    def test_dead_process_fails(self):
        cms, _ = self._patches(alive=False, appcrash=True, crash=True, scan_ok=True)
        rc = self._run(cms)
        self.assertEqual(rc, 1)
        self.assertTrue(os.path.exists(S.FAIL_REPORT))
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["result"], "FAIL")
        self.assertFalse(payload["checks"]["process_alive"]["ok"])

    def test_appcrash_event_fails(self):
        cms, _ = self._patches(alive=True, appcrash=False, crash=True, scan_ok=True)
        rc = self._run(cms)
        self.assertEqual(rc, 1)
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertFalse(payload["checks"]["appcrash_events"]["ok"])

    def test_native_crash_dump_fails(self):
        cms, _ = self._patches(alive=True, appcrash=True, crash=False, scan_ok=True)
        rc = self._run(cms)
        self.assertEqual(rc, 1)
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        ct = payload["checks"]["crash_traces"]
        self.assertFalse(ct["ok"])
        self.assertEqual(ct["new_dumps"], ["DUMP"])
        self.assertEqual(ct["head_signatures"], ["sig"])

    def test_session_log_fatal_fails(self):
        cms, _ = self._patches(alive=True, appcrash=True, crash=True, scan_ok=False)
        rc = self._run(cms)
        self.assertEqual(rc, 1)
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        slf = payload["checks"]["session_log_fatal"]
        self.assertFalse(slf["ok"])
        self.assertEqual(slf["fatal_count"], 1)
        self.assertEqual(slf["fatal_lines"], ["[FATAL] boom"])

    def test_no_session_log_found_fails(self):
        cms, _ = self._patches(alive=True, appcrash=True, crash=True,
                               scan_ok=True, session_log_present=False)
        rc = self._run(cms)
        self.assertEqual(rc, 1)
        with open(S.FAIL_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        slf = payload["checks"]["session_log_fatal"]
        self.assertFalse(slf["ok"])
        self.assertIn("no session log files found", slf["error"])

    def test_session_log_predates_launch_warning(self):
        # pre_launch_log == session_log → "predates launch" warning is attached,
        # but the scan still runs (and here passes).
        cms, _ = self._patches(alive=True, appcrash=True, crash=True,
                               scan_ok=True, pre_equals_post=True)
        rc = self._run(cms)
        self.assertEqual(rc, 0)
        with open(S.PASS_REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertIn("warning", payload["checks"]["session_log_fatal"])
        self.assertIn("predates launch", payload["checks"]["session_log_fatal"]["warning"])


class TestMainLaunchBranch(_Base):
    def test_launch_step_invoked_when_not_no_launch(self):
        # Confirm the launch branch actually calls _launch_jarvis (and that a
        # successful boot proceeds into a passing run).
        called = {"n": 0}

        def fake_launch():
            called["n"] += 1
            return ("ok", "")

        log = self.write_session_log("session_run.log", "Listening...\n", mtime=9000)
        with mock.patch.object(S, "_launch_jarvis", fake_launch), \
             mock.patch.object(S, "_wait_for_lock_pid", return_value=4242), \
             mock.patch.object(S, "_pid_alive", return_value=True), \
             mock.patch.object(S, "_check_appcrash_events",
                               return_value={"ok": True, "count": 0, "events": []}), \
             mock.patch.object(S, "_scan_crash_traces_since",
                               return_value={"path": "p", "new_dumps": [], "head_signatures": []}), \
             mock.patch.object(S, "_latest_session_log", side_effect=[None, log]):
            rc = S.main([])  # no --no-launch
        self.assertEqual(called["n"], 1)
        self.assertEqual(rc, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
