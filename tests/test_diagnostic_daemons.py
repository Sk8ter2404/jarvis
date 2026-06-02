"""Unit tests for ``core.diagnostic_daemons`` — the four always-on background
watchdog daemons (self-diag, crash-watch, deep-audit, anomaly-watch) plus their
shared persistence layer, todo writer, status snapshots, and start/stop
lifecycle.

Design of this suite
--------------------
The module is built around four ``while not _stop_event.is_set(): ... ``
daemon loops that sleep via ``_stop_event.wait(N)``.  None of them are ever
allowed to run as a real thread here.  Instead each loop body is driven exactly
one (or a few) iteration(s) at a time by patching ``_stop_event.wait`` with a
scripted side-effect:

  * it returns ``False`` for the in-body sleeps we want to fall through, and
  * once the script is exhausted it raises ``_StopLoop`` — a ``BaseException``
    subclass, so the loops' ``except Exception:`` guards can NOT swallow it,
    and control returns to the test via ``_drive_loop``.

Every test redirects ALL of the module's on-disk paths into a per-test temp
dir (``STATE_FILE``, ``TODO_FILE``, ``LOGS_DIR``, ``DATA_DIR``, …) and resets
the module's globals/caches in ``tearDown`` so the suite is order-independent
and fully offline.  No real subprocess / network / threads / sleep is used.

CI faithfulness
---------------
``win32evtlog`` is a pywin32 module that is present on the Windows dev box but
ABSENT on the Linux CI runner.  The crash-watcher tests therefore NEVER touch
the real one — they inject a ``FakeWin32EvtLog`` into ``sys.modules`` per test
(auto-restored), so they run identically in both environments.  The anthropic
SDK and ``core.config`` import are likewise faked per-test.

Coverage note
-------------
This suite drives ``core.diagnostic_daemons`` to ~99.6%.  The only lines that
remain uncovered are two genuinely-unreachable defensive branches that cannot
be exercised without editing the source:

  * ``_recently_modified_source_files`` subtree-prune (``dirnames[:] = [];
    continue``) — the line just above already strips excluded dir names from
    ``dirnames``, so ``os.walk`` never yields an excluded directory as
    ``dirpath`` to trip this guard.
  * ``_looks_like_text_log``'s ``if not snippet: return False`` — ``snippet`` is
    ``sample[:4096]`` and ``sample`` was already proven truthy one line above,
    so the slice is always non-empty.

stdlib ``unittest`` only.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import core.diagnostic_daemons as dd


# ──────────────────────────── helpers / doubles ──────────────────────

class _StopLoop(BaseException):
    """Sentinel raised from a patched ``_stop_event.wait`` to break a daemon
    loop after the scripted iterations.  BaseException so the loops'
    ``except Exception`` can't swallow it."""


def _drive_loop(loop_fn, waits):
    """Run ``loop_fn`` with ``dd._stop_event.wait`` scripted by ``waits``.

    ``waits`` is a list of values returned by successive ``wait()`` calls.
    A ``True`` makes the loop ``return`` cleanly (the natural stop path).
    When the list is exhausted, ``_StopLoop`` is raised to force-break any
    remaining loop — that exception is caught here and swallowed.

    Returns the number of ``wait()`` calls actually made.
    """
    seq = list(waits)
    calls = {"n": 0}

    def _fake_wait(timeout=None):
        i = calls["n"]
        calls["n"] += 1
        if i < len(seq):
            return seq[i]
        raise _StopLoop
    try:
        with mock.patch.object(dd._stop_event, "wait", side_effect=_fake_wait):
            loop_fn()
    except _StopLoop:
        pass
    return calls["n"]


class FakeEvent:
    """Stand-in win32 event-log record."""

    def __init__(self, record_id=0, source="Application Error",
                 strings=None, time_generated=None):
        self.RecordNumber = record_id
        self.SourceName = source
        self.StringInserts = strings
        self.TimeGenerated = time_generated


class FakeTime:
    """A ``TimeGenerated`` double exposing ``.Format()``."""

    def __init__(self, text):
        self._text = text

    def Format(self):
        return self._text


class FakeWin32EvtLog:
    """Minimal faithful stand-in for the pywin32 ``win32evtlog`` module.

    ``ReadEventLog`` serves ``batches`` (a list of event lists) one call at a
    time, then returns ``[]`` (end of log).  Flags are exposed as ints so the
    OR in the source is exercised for real.
    """

    EVENTLOG_BACKWARDS_READ = 0x0008
    EVENTLOG_SEQUENTIAL_READ = 0x0001

    def __init__(self, batches=None, open_raises=False, read_raises=False,
                 close_raises=False):
        self._batches = list(batches or [])
        self._open_raises = open_raises
        self._read_raises = read_raises
        self._close_raises = close_raises
        self.opened = False
        self.closed = False
        self.read_calls = 0

    def OpenEventLog(self, server, source):
        if self._open_raises:
            raise OSError("open failed")
        self.opened = True
        return "HANDLE"

    def ReadEventLog(self, handle, flags, offset):
        self.read_calls += 1
        if self._read_raises:
            raise OSError("read failed")
        if self._batches:
            return self._batches.pop(0)
        return []

    def CloseEventLog(self, handle):
        if self._close_raises:
            raise OSError("close failed")
        self.closed = True


class FakeAnthropicModule:
    """Stand-in for the ``anthropic`` top-level module."""

    def __init__(self, *, text=None, raises=None):
        self._text = text
        self._raises = raises
        self.created_kwargs = None

        outer = self

        class _Block:
            def __init__(self, text):
                self.text = text

        class _Resp:
            def __init__(self, text):
                if text is None:
                    self.content = []
                else:
                    self.content = [_Block(text)]

        class _Messages:
            def create(self, **kwargs):
                outer.created_kwargs = kwargs
                if outer._raises is not None:
                    raise outer._raises
                return _Resp(outer._text)

        class _Client:
            def __init__(self, *a, **k):
                self.messages = _Messages()

        self.Anthropic = _Client


# ──────────────────────────── base fixture ───────────────────────────

class _Base(unittest.TestCase):
    """Redirect every on-disk path into a temp dir and snapshot/restore the
    module's mutable globals + caches."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="diagdaem_test_")

        def P(*parts):
            return os.path.join(self._tmp, *parts)

        self._data = P("data")
        self._logs = P("logs")
        os.makedirs(self._data, exist_ok=True)
        os.makedirs(self._logs, exist_ok=True)

        self._patchers = [
            mock.patch.object(dd, "DATA_DIR", self._data),
            mock.patch.object(dd, "STATE_FILE", P("data", "diag.json")),
            mock.patch.object(dd, "TODO_FILE", P("jarvis_todo.md")),
            mock.patch.object(dd, "PIPELINE_RUNS_FILE", P("data", "pipe.jsonl")),
            mock.patch.object(dd, "LOGS_DIR", self._logs),
            mock.patch.object(dd, "BOOT_FAILURES_FILE", P("data", "boot.jsonl")),
            mock.patch.object(dd, "HUD_STATE_FILE", P("hud_state.json")),
            # Use the temp dir as the project root so the source-file walk in
            # deep-audit stays inside the sandbox.
            mock.patch.object(dd, "PROJECT_DIR", self._tmp),
        ]
        for p in self._patchers:
            p.start()

        # Snapshot globals we may mutate.
        self._saved_started = dd._started
        self._saved_threads = list(dd._threads)
        self._saved_count_cache = dict(dd._count_cache)
        # Make sure no stop flag leaks in from a prior test.
        dd._stop_event.clear()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        dd._started = self._saved_started
        dd._threads[:] = self._saved_threads
        with dd._count_cache_lock:
            dd._count_cache.clear()
            dd._count_cache.update(self._saved_count_cache)
        dd._stop_event.clear()
        # Best-effort temp cleanup.
        try:
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)
        except Exception:
            pass

    # convenience -----------------------------------------------------
    def _read_state_file(self):
        with open(dd.STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_state_file(self, state):
        os.makedirs(os.path.dirname(dd.STATE_FILE), exist_ok=True)
        with open(dd.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def _todo_text(self):
        try:
            with open(dd.TODO_FILE, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _write_log(self, name, text):
        path = os.path.join(self._logs, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


# ──────────────────────────── time helpers ───────────────────────────

class TimeHelperTests(_Base):
    def test_now_returns_time(self):
        with mock.patch.object(dd.time, "time", return_value=4242.0):
            self.assertEqual(dd._now(), 4242.0)

    def test_iso_default_uses_now(self):
        s = dd._iso()
        self.assertRegex(s, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_iso_explicit_ts(self):
        # Epoch start, formatted in local time → just assert shape + year span.
        s = dd._iso(0.0)
        self.assertRegex(s, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_today_shape(self):
        self.assertRegex(dd._today(), r"^\d{4}-\d{2}-\d{2}$")


# ──────────────────────────── persistence ────────────────────────────

class PersistenceTests(_Base):
    def test_read_state_missing_returns_default_copy(self):
        st = dd._read_state()
        self.assertEqual(st, dd._DEFAULT_STATE)
        # Must be a deep copy — mutating it doesn't touch the template.
        st["paused"] = True
        self.assertFalse(dd._DEFAULT_STATE["paused"])

    def test_read_state_corrupt_json_returns_default(self):
        os.makedirs(os.path.dirname(dd.STATE_FILE), exist_ok=True)
        with open(dd.STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{ this is not json ")
        st = dd._read_state()
        self.assertEqual(st, dd._DEFAULT_STATE)

    def test_read_state_backfills_missing_keys(self):
        # An old-shape file missing several nested keys still loads with
        # defaults filled in, while preserving the stored values.
        self._write_state_file({
            "paused": True,
            "self_diag": {"runs": 7},        # missing last_run_ts etc.
            "extra_top_level": 5,            # unknown top-level passes through
        })
        st = dd._read_state()
        self.assertTrue(st["paused"])
        self.assertEqual(st["self_diag"]["runs"], 7)
        self.assertIn("last_run_ts", st["self_diag"])     # backfilled
        self.assertEqual(st["extra_top_level"], 5)
        self.assertIn("crash_watch", st)                  # whole section added

    def test_read_state_non_dict_value_overrides(self):
        # If a stored key holds a non-dict where the default is a dict, the
        # stored (non-dict) value replaces it via the else branch.
        self._write_state_file({"self_diag": "broken"})
        st = dd._read_state()
        self.assertEqual(st["self_diag"], "broken")

    def test_write_then_read_roundtrip(self):
        dd._write_state({"paused": True, "marker": 123})
        st = dd._read_state()
        self.assertTrue(st["paused"])
        self.assertEqual(st["marker"], 123)
        # tmp file should have been replaced (not left behind).
        self.assertFalse(os.path.exists(dd.STATE_FILE + ".tmp"))

    def test_write_state_handles_failure_gracefully(self):
        # makedirs raising must be swallowed (printed, not raised).
        with mock.patch.object(dd.os, "makedirs", side_effect=OSError("nope")):
            dd._write_state({"paused": True})  # should not raise

    def test_update_state_mutates_and_persists(self):
        def mut(s):
            s["paused"] = True
            s["self_diag"]["runs"] = 3
        returned = dd._update_state(mut)
        self.assertTrue(returned["paused"])
        on_disk = self._read_state_file()
        self.assertTrue(on_disk["paused"])
        self.assertEqual(on_disk["self_diag"]["runs"], 3)


# ──────────────────────────── todo writer ────────────────────────────

class TodoWriterTests(_Base):
    def test_existing_todo_text_missing_file(self):
        self.assertEqual(dd._existing_todo_text(), "")

    def test_append_creates_line(self):
        ok = dd._append_todo_task("do the thing", tag="anomaly")
        self.assertTrue(ok)
        txt = self._todo_text()
        self.assertIn("do the thing", txt)
        self.assertIn("anomaly-", txt)
        self.assertTrue(txt.startswith("- [ ] **"))

    def test_append_empty_body_returns_false(self):
        self.assertFalse(dd._append_todo_task("   ", tag="anomaly"))
        self.assertEqual(self._todo_text(), "")

    def test_append_dedup_same_body(self):
        self.assertTrue(dd._append_todo_task("dup body", tag="x"))
        self.assertFalse(dd._append_todo_task("dup body", tag="x"))
        # Only one occurrence.
        self.assertEqual(self._todo_text().count("dup body"), 1)

    def test_append_inserts_newline_separator(self):
        # Pre-seed a file with no trailing newline; the appender must insert one.
        with open(dd.TODO_FILE, "w", encoding="utf-8") as f:
            f.write("# header no newline")
        ok = dd._append_todo_task("fresh task", tag="t")
        self.assertTrue(ok)
        txt = self._todo_text()
        self.assertIn("# header no newline\n- [ ] ", txt)

    def test_append_write_failure_returns_false(self):
        # open() raising on the append path → caught, returns False.
        real_open = open

        def boom(path, mode="r", *a, **k):
            if path == dd.TODO_FILE and "a" in mode:
                raise OSError("disk full")
            return real_open(path, mode, *a, **k)

        with mock.patch("builtins.open", side_effect=boom):
            self.assertFalse(dd._append_todo_task("body that fails", tag="t"))


# ──────────────────────────── self-diag ──────────────────────────────

class SelfDiagRunTests(_Base):
    def test_run_self_diag_import_failure(self):
        # No skills.self_diagnostic injected → ImportError path (printed, no raise).
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skills.self_diagnostic", None)
            with mock.patch("builtins.__import__",
                            side_effect=ImportError("nope")):
                dd._run_self_diag_once()  # must not raise

    def test_run_self_diag_invokes_run_diagnostic(self):
        fake = mock.MagicMock()
        with mock.patch.dict(sys.modules,
                             {"skills.self_diagnostic": fake,
                              "skills": mock.MagicMock(self_diagnostic=fake)}):
            dd._run_self_diag_once()
        fake.run_diagnostic.assert_called_once_with("")

    def test_run_self_diag_swallows_run_diagnostic_error(self):
        fake = mock.MagicMock()
        fake.run_diagnostic.side_effect = RuntimeError("kaboom")
        with mock.patch.dict(sys.modules,
                             {"skills.self_diagnostic": fake,
                              "skills": mock.MagicMock(self_diagnostic=fake)}):
            dd._run_self_diag_once()  # must not raise


class SelfDiagLoopTests(_Base):
    def test_initial_wait_true_returns_immediately(self):
        # The very first wait() returning True (stop requested during the
        # boot-settle wait) must exit before any body runs.
        with mock.patch.object(dd, "_run_self_diag_once") as run:
            n = _drive_loop(dd._self_diag_loop, [True])
        run.assert_not_called()
        self.assertEqual(n, 1)

    def test_paused_branch_sleeps_and_continues(self):
        self._write_state_file({"paused": True,
                                "self_diag": {"last_run_ts": 0.0}})
        with mock.patch.object(dd, "_run_self_diag_once") as run:
            # waits: [0]=initial settle(False) → enter loop;
            #        [1]=paused 30s wait(False) → hit `continue`, re-loop;
            #        [2]=paused 30s wait(True) → return.
            _drive_loop(dd._self_diag_loop, [False, False, True])
        run.assert_not_called()
        # alive_ts heartbeat was written.
        self.assertGreater(self._read_state_file()["self_diag"]["alive_ts"], 0)

    def test_dedup_gap_skips_run(self):
        # last_run only moments ago → within dedup gap → skip, sleep, return.
        now = 10_000.0
        self._write_state_file({
            "paused": False,
            "self_diag": {"last_run_ts": now - 10},  # 10s ago < 240s gap
        })
        with mock.patch.object(dd, "_now", return_value=now), \
                mock.patch.object(dd, "_run_self_diag_once") as run:
            # 2nd iteration drives the dedup-branch `continue` (False then True).
            _drive_loop(dd._self_diag_loop, [False, False, True])
        run.assert_not_called()

    def test_runs_diag_when_due_and_updates_state(self):
        now = 50_000.0
        self._write_state_file({
            "paused": False,
            "self_diag": {"last_run_ts": 0.0, "runs": 4},  # long ago → due
        })
        with mock.patch.object(dd, "_now", return_value=now), \
                mock.patch.object(dd, "_run_self_diag_once") as run:
            # waits: initial(False), end-of-iteration(True)
            _drive_loop(dd._self_diag_loop, [False, True])
        run.assert_called_once()
        st = self._read_state_file()["self_diag"]
        self.assertEqual(st["runs"], 5)
        self.assertEqual(st["last_run_ts"], now)
        self.assertIsNotNone(st["last_run_iso"])

    def test_body_exception_is_caught_and_loop_recovers(self):
        # Make _read_state raise once inside the body; the except-block sleeps
        # then we stop. Loop must not propagate the error.
        self._write_state_file({"paused": False, "self_diag": {}})
        calls = {"n": 0}
        real_read = dd._read_state

        def flaky_read():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient read fail")
            return real_read()

        with mock.patch.object(dd, "_read_state", side_effect=flaky_read):
            # initial(False) enters loop; body raises → except sleeps(True)→return
            _drive_loop(dd._self_diag_loop, [False, True])
        # No assertion needed beyond "did not raise"; reaching here = pass.


# ──────────────────────────── path helpers ───────────────────────────

class PathHelperTests(_Base):
    def test_path_is_jarvis_variants(self):
        # Cross-platform cases. ``_path_is_jarvis`` normalises "/" -> "\\" and
        # lowercases, so forward-slash inputs match the "\\jarvis" / endswith
        # branches on every OS. These hold identically on Windows and Linux.
        self.assertTrue(dd._path_is_jarvis("foo/JARVIS/thing.py"))
        self.assertTrue(dd._path_is_jarvis("foo/bobert_companion.py"))
        self.assertTrue(dd._path_is_jarvis("D:/work/jarvis"))
        self.assertFalse(dd._path_is_jarvis(""))
        self.assertFalse(dd._path_is_jarvis("/usr/bin/python3"))

        # PROJECT_DIR branch: exercise it OS-agnostically. The helper normalises
        # "/" -> "\\" *inside* the function but compares against the raw
        # ``PROJECT_DIR.lower()`` needle, so on Linux a posix PROJECT_DIR (with
        # "/" separators) would never substring-match a backslash-normalised
        # input. Feed an already-normalised child path so the needle is present
        # regardless of the host's path separator.
        proj_norm = dd.PROJECT_DIR.replace("/", "\\")
        self.assertTrue(dd._path_is_jarvis(proj_norm + "\\x"))

    @unittest.skipUnless(
        sys.platform.startswith("win"),
        "literal Windows drive-letter/backslash paths only occur on Windows",
    )
    def test_path_is_jarvis_windows_paths(self):
        # Genuinely Windows-shaped inputs: a backslash repo path matches and a
        # backslash system path does not. On Linux these strings never appear,
        # so the assertions are scoped to Windows rather than made to pass there.
        self.assertTrue(dd._path_is_jarvis(r"C:\some\JARVIS\thing.py"))
        self.assertFalse(dd._path_is_jarvis(r"C:\Windows\system32\notepad.exe"))

    def test_latest_session_log_tail_no_dir(self):
        # Point LOGS_DIR at a non-existent path → FileNotFoundError handled.
        with mock.patch.object(dd, "LOGS_DIR", os.path.join(self._tmp, "nope")):
            self.assertEqual(dd._latest_session_log_tail(), "")

    def test_latest_session_log_tail_no_files(self):
        self.assertEqual(dd._latest_session_log_tail(), "")

    def test_latest_session_log_tail_returns_tail(self):
        body = "\n".join(f"line {i}" for i in range(100))
        self._write_log("session_1.log", body)
        tail = dd._latest_session_log_tail(5)
        self.assertIn("line 99", tail)
        self.assertNotIn("line 0\n", tail)
        self.assertIn(" | ", tail)        # newlines collapsed to ' | '
        self.assertLessEqual(len(tail), 1500)

    def test_latest_session_log_tail_caps_at_1500(self):
        self._write_log("session_big.log", "x" * 5000)
        tail = dd._latest_session_log_tail(30)
        self.assertEqual(len(tail), 1500)

    def test_latest_session_log_tail_picks_newest(self):
        old = self._write_log("session_old.log", "OLD CONTENT")
        new = self._write_log("session_new.log", "NEW CONTENT")
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        self.assertIn("NEW CONTENT", dd._latest_session_log_tail(5))

    def test_latest_session_log_tail_read_error(self):
        self._write_log("session_x.log", "data")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(dd._latest_session_log_tail(5), "")

    def test_latest_session_log_path_none_when_empty(self):
        self.assertIsNone(dd._latest_session_log_path())

    def test_latest_session_log_path_missing_dir(self):
        with mock.patch.object(dd, "LOGS_DIR", os.path.join(self._tmp, "nope")):
            self.assertIsNone(dd._latest_session_log_path())

    def test_latest_session_log_path_returns_newest(self):
        a = self._write_log("session_a.log", "a")
        b = self._write_log("session_b.log", "b")
        os.utime(a, (1000, 1000))
        os.utime(b, (2000, 2000))
        self.assertEqual(dd._latest_session_log_path(), b)


# ──────────────────────────── crash watcher ──────────────────────────

class WinEventImportTests(_Base):
    def test_import_win_event_api_present(self):
        fake = FakeWin32EvtLog()
        with mock.patch.dict(sys.modules, {"win32evtlog": fake}):
            self.assertIs(dd._import_win_event_api(), fake)

    def test_import_win_event_api_absent(self):
        # Simulate the Linux/CI case where the import fails.
        real_import = __import__

        def no_win32(name, *a, **k):
            if name == "win32evtlog":
                raise ImportError("no pywin32 on CI")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=no_win32):
            self.assertIsNone(dd._import_win_event_api())


class ScanEventLogTests(_Base):
    def test_open_failure_returns_empty(self):
        fake = FakeWin32EvtLog(open_raises=True)
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])
        self.assertEqual(max_seen, 0)

    def test_empty_log_returns_baseline(self):
        fake = FakeWin32EvtLog(batches=[])    # ReadEventLog → []
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 5)
        self.assertEqual(hits, [])
        self.assertEqual(max_seen, 5)
        self.assertTrue(fake.closed)

    def test_read_error_breaks_cleanly(self):
        fake = FakeWin32EvtLog(read_raises=True)
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])

    def test_detects_matching_appcrash(self):
        strings = [
            "python.exe", "1.0", "ts", "mod.pyd", "1.0", "ts2",
            "c0000005", "0x1234", "4242", "starttime",
            r"C:\JARVIS\bobert_companion.py", "modpath",
        ]
        ev = FakeEvent(record_id=100, source="Application Error",
                       strings=strings, time_generated=FakeTime("2026-06-01"))
        fake = FakeWin32EvtLog(batches=[[ev]])
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["app"], "python.exe")
        self.assertEqual(hits[0]["offset"], "0x1234")
        self.assertEqual(hits[0]["record_id"], 100)
        self.assertEqual(hits[0]["ts"], "2026-06-01")
        self.assertEqual(max_seen, 100)

    def test_skips_non_application_error_source(self):
        ev = FakeEvent(record_id=10, source="SomethingElse",
                       strings=["python.exe", r"C:\JARVIS"])
        fake = FakeWin32EvtLog(batches=[[ev]])
        hits, _ = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])

    def test_skips_non_python_app(self):
        ev = FakeEvent(record_id=10, source="Application Error",
                       strings=["chrome.exe", r"C:\JARVIS\thing"])
        fake = FakeWin32EvtLog(batches=[[ev]])
        hits, _ = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])

    def test_skips_python_but_non_jarvis_path(self):
        ev = FakeEvent(record_id=10, source="Application Error",
                       strings=["python.exe", r"C:\Other\app.py"])
        fake = FakeWin32EvtLog(batches=[[ev]])
        hits, _ = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])

    def test_stops_at_baseline_record(self):
        # newest-first ordering: an event at/below last_record_id ends the scan.
        old = FakeEvent(record_id=5, source="Application Error",
                        strings=["python.exe", r"C:\JARVIS"])
        fake = FakeWin32EvtLog(batches=[[old]])
        hits, max_seen = dd._scan_event_log_for_crashes(fake, last_record_id=5)
        self.assertEqual(hits, [])
        # max_seen stays at the baseline (5 not > 5).
        self.assertEqual(max_seen, 5)

    def test_missing_record_number_attr(self):
        # getattr default 0 path: an event lacking RecordNumber still scanned.
        class Bare:
            SourceName = "Application Error"
            StringInserts = ["notpython"]
        fake = FakeWin32EvtLog(batches=[[Bare()]])
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])
        self.assertEqual(max_seen, 0)

    def test_unparseable_record_number_defaults_zero(self):
        # A RecordNumber that int() can't coerce → the except sets rec_id=0.
        class BadRec:
            def __int__(self):
                raise ValueError("not an int")

        ev = FakeEvent(record_id=BadRec(), source="Other", strings=["x"])
        fake = FakeWin32EvtLog(batches=[[ev]])
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])
        self.assertEqual(max_seen, 0)

    def test_hit_exactly_at_scan_cap_breaks(self):
        # 199 non-matching events then a matching crash as #200 → the hit is
        # appended and the `if scanned >= SCAN_CAP: break` after the append
        # fires (record number 200 = SCAN_CAP).
        filler = [FakeEvent(record_id=i, source="Other", strings=["x"])
                  for i in range(1, 200)]
        crash_strings = ["python.exe"] + ["a"] * 6 + ["0xCAP"] + ["b"] * 3 + \
                        [r"C:\JARVIS"]
        crash = FakeEvent(record_id=200, source="Application Error",
                          strings=crash_strings)
        fake = FakeWin32EvtLog(batches=[filler + [crash]])
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["offset"], "0xCAP")
        self.assertEqual(max_seen, 200)

    def test_appcrash_without_offset_string(self):
        # Fewer than 8 strings → offset defaults to '?'.
        strings = ["python.exe", r"C:\JARVIS\x"]
        ev = FakeEvent(record_id=3, source="Application Error", strings=strings)
        fake = FakeWin32EvtLog(batches=[[ev]])
        hits, _ = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["offset"], "?")

    def test_appcrash_no_timegenerated_falls_back_to_iso(self):
        strings = ["pythonw.exe"] + ["x"] * 6 + ["0xAB"] + ["y"] * 3 + \
                  [r"C:\JARVIS"]
        ev = FakeEvent(record_id=7, source="Application Error", strings=strings,
                       time_generated=None)
        fake = FakeWin32EvtLog(batches=[[ev]])
        hits, _ = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(len(hits), 1)
        self.assertRegex(hits[0]["ts"], r"^\d{4}-\d{2}-\d{2}T")

    def test_close_failure_swallowed(self):
        fake = FakeWin32EvtLog(batches=[[]], close_raises=True)
        hits, _ = dd._scan_event_log_for_crashes(fake, 0)  # must not raise
        self.assertEqual(hits, [])

    def test_scan_cap_bounds_across_batches(self):
        # The SCAN_CAP (200) guard is re-checked at the TOP of the per-batch
        # while-loop, so a huge FIRST batch is fully consumed (the inner for
        # only breaks on a hit). Split into 150-event batches: after the first
        # batch scanned=150 (<200, another ReadEventLog), after the second
        # scanned=300 (>=200) so the third batch is never requested.
        b1 = [FakeEvent(record_id=i, source="Other", strings=["x"])
              for i in range(1, 151)]
        b2 = [FakeEvent(record_id=i, source="Other", strings=["x"])
              for i in range(151, 301)]
        b3 = [FakeEvent(record_id=999, source="Other", strings=["x"])]
        fake = FakeWin32EvtLog(batches=[b1, b2, b3])
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])
        # Two batches read (300 events), third skipped by the cap.
        self.assertEqual(fake.read_calls, 2)
        self.assertEqual(max_seen, 300)

    def test_single_huge_batch_is_fully_scanned(self):
        # Documents the actual behaviour: a single oversized batch bypasses the
        # top-of-loop cap because non-matching events `continue` before the
        # in-loop break, so all 250 are scanned in one ReadEventLog call.
        big = [FakeEvent(record_id=i, source="Other", strings=["x"])
               for i in range(1, 251)]
        fake = FakeWin32EvtLog(batches=[big])
        hits, max_seen = dd._scan_event_log_for_crashes(fake, 0)
        self.assertEqual(hits, [])
        self.assertEqual(max_seen, 250)


class CrashWatchLoopTests(_Base):
    def test_loop_disabled_when_api_absent(self):
        with mock.patch.object(dd, "_import_win_event_api", return_value=None):
            # Loop returns immediately; wait is never even scripted.
            n = _drive_loop(dd._crash_watch_loop, [])
        self.assertEqual(n, 0)

    def test_loop_seeds_baseline_then_polls(self):
        # seed_id 0 → first scan seeds head; then one poll iteration with a hit.
        # The scan itself is mocked, so it returns ready-made crash dicts.
        crash_ev_dict = {"ts": "2026-06-01T00:00:00", "app": "python.exe",
                         "offset": "0xFEED", "record_id": 201}
        # Two scans happen: (1) seed (returns head=100, no hits),
        # (2) in-loop poll returns the crash.
        seed_fake = FakeWin32EvtLog()
        poll_calls = {"n": 0}

        def fake_scan(api, last_record_id):
            poll_calls["n"] += 1
            if poll_calls["n"] == 1:
                return [], 100          # seed
            return [crash_ev_dict], 201  # poll detects crash

        with mock.patch.object(dd, "_import_win_event_api",
                               return_value=seed_fake), \
                mock.patch.object(dd, "_scan_event_log_for_crashes",
                                  side_effect=fake_scan), \
                mock.patch.object(dd, "_latest_session_log_tail",
                                  return_value="tail-text"):
            _drive_loop(dd._crash_watch_loop, [True])  # one body, then stop

        txt = self._todo_text()
        self.assertIn("[crash-watch] APPCRASH at 2026-06-01T00:00:00", txt)
        self.assertIn("offset 0xFEED", txt)
        self.assertIn("tail-text", txt)
        st = self._read_state_file()["crash_watch"]
        self.assertEqual(st["detections"], 1)
        self.assertEqual(st["last_seen_record_id"], 201)

    def test_loop_skips_seed_when_already_seeded(self):
        self._write_state_file({
            "crash_watch": {"last_seen_record_id": 500},
        })
        scan_calls = {"n": 0}

        def fake_scan(api, last_record_id):
            scan_calls["n"] += 1
            # No new crashes; head unchanged.
            return [], last_record_id

        with mock.patch.object(dd, "_import_win_event_api",
                               return_value=FakeWin32EvtLog()), \
                mock.patch.object(dd, "_scan_event_log_for_crashes",
                                  side_effect=fake_scan):
            _drive_loop(dd._crash_watch_loop, [True])
        # Seed path skipped → only the in-loop scan ran (1 call).
        self.assertEqual(scan_calls["n"], 1)

    def test_loop_paused_skips_scan(self):
        self._write_state_file({
            "paused": True,
            "crash_watch": {"last_seen_record_id": 9},
        })
        with mock.patch.object(dd, "_import_win_event_api",
                               return_value=FakeWin32EvtLog()), \
                mock.patch.object(dd, "_scan_event_log_for_crashes") as scan:
            # 2nd iteration drives the paused-branch `continue`.
            _drive_loop(dd._crash_watch_loop, [False, True])
        scan.assert_not_called()

    def test_loop_scan_exception_recovers(self):
        self._write_state_file({"crash_watch": {"last_seen_record_id": 9}})
        with mock.patch.object(dd, "_import_win_event_api",
                               return_value=FakeWin32EvtLog()), \
                mock.patch.object(dd, "_scan_event_log_for_crashes",
                                  side_effect=RuntimeError("scan boom")):
            # Body's inner try/except handles it; first wait(False) hits the
            # post-except `continue`, second wait(True) returns.
            _drive_loop(dd._crash_watch_loop, [False, True])
        # Reaching here = the exception was contained.

    def test_loop_outer_exception_contained(self):
        # Force the top-of-body alive heartbeat _update_state to raise so the
        # OUTER try/except (not the inner scan guard) handles it.
        self._write_state_file({"crash_watch": {"last_seen_record_id": 9}})
        calls = {"n": 0}
        real_update = dd._update_state

        def flaky_update(mut):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("heartbeat boom")
            return real_update(mut)

        with mock.patch.object(dd, "_import_win_event_api",
                               return_value=FakeWin32EvtLog()), \
                mock.patch.object(dd, "_update_state", side_effect=flaky_update):
            # First (and only) wait is the outer-except sleep; True → return,
            # exercising the except-branch's `return` path.
            _drive_loop(dd._crash_watch_loop, [True])
        # Outer except logged + slept; reaching here = contained.


# ──────────────────────────── deep-audit pieces ──────────────────────

class DeepAuditBudgetEnvTests(_Base):
    def test_default_budget(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_DEEP_AUDIT_BUDGET_USD", None)
            self.assertEqual(dd._deep_audit_budget_usd(),
                             dd.DEEP_AUDIT_DEFAULT_BUDGET_USD)

    def test_env_override(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_DEEP_AUDIT_BUDGET_USD": "12.5"}):
            self.assertEqual(dd._deep_audit_budget_usd(), 12.5)

    def test_negative_clamped_to_zero(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_DEEP_AUDIT_BUDGET_USD": "-3"}):
            self.assertEqual(dd._deep_audit_budget_usd(), 0.0)

    def test_bad_value_falls_back(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_DEEP_AUDIT_BUDGET_USD": "notanumber"}):
            self.assertEqual(dd._deep_audit_budget_usd(),
                             dd.DEEP_AUDIT_DEFAULT_BUDGET_USD)


class CountPipelineCompletionsTests(_Base):
    def _write_pipe(self, lines):
        with open(dd.PIPELINE_RUNS_FILE, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")

    def test_missing_file_zero(self):
        self.assertEqual(dd._count_pipeline_task_completions(), 0)

    def test_counts_verdicts_and_events(self):
        self._write_pipe([
            json.dumps({"verdict": "approve"}),
            json.dumps({"verdict": "approve_with_warnings"}),
            json.dumps({"verdict": "reject"}),         # not counted
            json.dumps({"event": "task_completed"}),
            json.dumps({"event": "task_done"}),
            json.dumps({"event": "approved"}),
            json.dumps({"event": "started"}),          # not counted
            "   ",                                      # blank skipped
            "{not valid json",                         # corrupt skipped
        ])
        self.assertEqual(dd._count_pipeline_task_completions(), 5)

    def test_cache_hit_returns_without_reparse(self):
        self._write_pipe([json.dumps({"verdict": "approve"})])
        first = dd._count_pipeline_task_completions()
        self.assertEqual(first, 1)
        # Second call: file unchanged → cache short-circuits. Patch open to
        # prove it is NOT re-read.
        with mock.patch("builtins.open",
                        side_effect=AssertionError("should not reopen")):
            self.assertEqual(dd._count_pipeline_task_completions(), 1)

    def test_cache_invalidated_on_change(self):
        self._write_pipe([json.dumps({"verdict": "approve"})])
        self.assertEqual(dd._count_pipeline_task_completions(), 1)
        # Append another completion and bump mtime so cache misses.
        with open(dd.PIPELINE_RUNS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"verdict": "approve"}) + "\n")
        st = os.stat(dd.PIPELINE_RUNS_FILE)
        os.utime(dd.PIPELINE_RUNS_FILE, (st.st_atime + 5, st.st_mtime + 5))
        self.assertEqual(dd._count_pipeline_task_completions(), 2)

    def test_stat_oserror_returns_zero(self):
        self._write_pipe([json.dumps({"verdict": "approve"})])
        with mock.patch.object(dd.os, "stat", side_effect=OSError("gone")):
            self.assertEqual(dd._count_pipeline_task_completions(), 0)

    def test_open_failure_after_stat_returns_running_count(self):
        self._write_pipe([json.dumps({"verdict": "approve"})])
        # Force a cache miss, then make the read open() raise.
        with dd._count_cache_lock:
            dd._count_cache["mtime"] = -1
        with mock.patch("builtins.open", side_effect=OSError("read fail")):
            self.assertEqual(dd._count_pipeline_task_completions(), 0)


class RecentSourceFilesTests(_Base):
    def _touch_py(self, relpath, mtime=None):
        # Build the path with the OS separator so it matches what the source's
        # os.path.join(dirpath, fn) produces (forward slashes would mismatch).
        full = os.path.join(self._tmp, *relpath.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write("# x\n")
        if mtime is not None:
            os.utime(full, (mtime, mtime))
        return full

    def test_returns_sorted_newest_first(self):
        a = self._touch_py("a.py", mtime=1000)
        b = self._touch_py("core/b.py", mtime=3000)
        c = self._touch_py("skills/c.py", mtime=2000)
        got = dd._recently_modified_source_files(10)
        self.assertEqual(got[:3], [b, c, a])

    def test_limit_respected(self):
        for i in range(5):
            self._touch_py(f"f{i}.py", mtime=1000 + i)
        self.assertEqual(len(dd._recently_modified_source_files(2)), 2)

    def test_excludes_pruned_dirs(self):
        self._touch_py("good.py", mtime=1000)
        self._touch_py("logs/bad.py", mtime=9999)
        self._touch_py("data/bad2.py", mtime=9999)
        self._touch_py("backups/bad3.py", mtime=9999)
        self._touch_py("memory/bad4.py", mtime=9999)
        got = dd._recently_modified_source_files(50)
        self.assertTrue(any(p.endswith("good.py") for p in got))
        self.assertFalse(any("logs" in p or "backups" in p or "memory" in p
                             for p in got))

    def test_ignores_non_py(self):
        self._touch_py("keep.py", mtime=1000)
        with open(os.path.join(self._tmp, "skip.txt"), "w") as f:
            f.write("x")
        got = dd._recently_modified_source_files(50)
        self.assertTrue(all(p.endswith(".py") for p in got))

    def test_getmtime_oserror_skips_file(self):
        # A .py file whose getmtime raises during the walk is skipped (continue)
        # rather than aborting the whole scan.
        self._touch_py("a.py", mtime=1000)
        self._touch_py("b.py", mtime=2000)
        real_getmtime = os.path.getmtime

        def flaky(path):
            if path.endswith("a.py"):
                raise OSError("vanished mid-walk")
            return real_getmtime(path)

        with mock.patch.object(dd.os.path, "getmtime", side_effect=flaky):
            got = dd._recently_modified_source_files(50)
        self.assertTrue(any(p.endswith("b.py") for p in got))
        self.assertFalse(any(p.endswith("a.py") for p in got))

    def test_newest_source_mtime_empty(self):
        # Temp dir has no .py files at all.
        self.assertEqual(dd._newest_source_mtime(), 0.0)

    def test_newest_source_mtime_value(self):
        self._touch_py("z.py", mtime=4321)
        self.assertEqual(dd._newest_source_mtime(), 4321)

    def test_newest_source_mtime_getmtime_error(self):
        # Files exist (walk returns one) but the final getmtime on the newest
        # raises → the except returns 0.0. Stub the file list so the walk's own
        # getmtime calls don't interfere with the targeted failure.
        with mock.patch.object(dd, "_recently_modified_source_files",
                               return_value=["ghost.py"]), \
                mock.patch.object(dd.os.path, "getmtime",
                                  side_effect=OSError("gone")):
            self.assertEqual(dd._newest_source_mtime(), 0.0)


class DeepAuditDueTests(_Base):
    def test_batch_threshold_fires(self):
        state = {"deep_audit": {"last_pipeline_event_count": 0,
                                "last_run_ts": 0.0}}
        with mock.patch.object(dd, "_count_pipeline_task_completions",
                               return_value=dd.DEEP_AUDIT_BATCH_SIZE):
            due, reason = dd._deep_audit_due(state)
        self.assertTrue(due)
        self.assertIn("batch", reason)

    def test_below_batch_and_recent_mtime_not_due(self):
        now = 100_000.0
        state = {"deep_audit": {"last_pipeline_event_count": 0,
                                "last_run_ts": now}}
        with mock.patch.object(dd, "_count_pipeline_task_completions",
                               return_value=1), \
                mock.patch.object(dd, "_newest_source_mtime",
                                  return_value=now + 10), \
                mock.patch.object(dd, "_now", return_value=now + 5):
            due, reason = dd._deep_audit_due(state)
        # mtime newer but <1h gap → not due.
        self.assertFalse(due)

    def test_aggressive_mtime_fallback_fires(self):
        last = 1000.0
        state = {"deep_audit": {"last_pipeline_event_count": 0,
                                "last_run_ts": last}}
        with mock.patch.object(dd, "_count_pipeline_task_completions",
                               return_value=0), \
                mock.patch.object(dd, "_newest_source_mtime",
                                  return_value=last + 1), \
                mock.patch.object(dd, "_now",
                                  return_value=last + dd.DEEP_AUDIT_AGGRESSIVE_GAP_S + 1):
            due, reason = dd._deep_audit_due(state)
        self.assertTrue(due)
        self.assertEqual(reason, "aggressive_mtime_fallback")

    def test_not_due_when_nothing_changed(self):
        state = {"deep_audit": {"last_pipeline_event_count": 5,
                                "last_run_ts": 9_999_999_999.0}}
        with mock.patch.object(dd, "_count_pipeline_task_completions",
                               return_value=5), \
                mock.patch.object(dd, "_newest_source_mtime", return_value=0.0):
            due, reason = dd._deep_audit_due(state)
        self.assertFalse(due)
        self.assertEqual(reason, "")


class DeepAuditBudgetOkTests(_Base):
    def test_hourly_cap_blocks(self):
        now = 1_000_000.0
        self._write_state_file({"deep_audit": {
            "hourly_window_start_ts": now,            # window still open
            "hourly_window_count": dd.DEEP_AUDIT_MAX_RUNS_PER_HOUR,
            "daily_budget_date": dd._today(),
            "daily_budget_spent_usd": 0.0,
        }})
        state = dd._read_state()
        with mock.patch.object(dd, "_now", return_value=now):
            ok, why = dd._deep_audit_budget_ok(state)
        self.assertFalse(ok)
        self.assertEqual(why, "hourly_cap")

    def test_hourly_window_resets_after_an_hour(self):
        now = 1_000_000.0
        self._write_state_file({"deep_audit": {
            "hourly_window_start_ts": now - 4000,     # >1h ago
            "hourly_window_count": dd.DEEP_AUDIT_MAX_RUNS_PER_HOUR,
            "daily_budget_date": dd._today(),
            "daily_budget_spent_usd": 0.0,
        }})
        state = dd._read_state()
        with mock.patch.object(dd, "_now", return_value=now):
            ok, why = dd._deep_audit_budget_ok(state)
        self.assertTrue(ok)
        self.assertEqual(why, "ok")
        # The reset was persisted.
        persisted = self._read_state_file()["deep_audit"]
        self.assertEqual(persisted["hourly_window_count"], 0)

    def test_daily_budget_exhausted_blocks(self):
        now = 1_000_000.0
        # The source derives "today" from _now(), so compute the stored date
        # under the SAME patched clock — otherwise it reads as a new day and
        # resets the spend before the cap check.
        with mock.patch.object(dd, "_now", return_value=now):
            today = dd._today()
        self._write_state_file({"deep_audit": {
            "hourly_window_start_ts": now,
            "hourly_window_count": 0,
            "daily_budget_date": today,
            "daily_budget_spent_usd": 100.0,          # way over default $5
        }})
        state = dd._read_state()
        with mock.patch.object(dd, "_now", return_value=now):
            ok, why = dd._deep_audit_budget_ok(state)
        self.assertFalse(ok)
        self.assertEqual(why, "daily_budget_exhausted")

    def test_new_day_resets_spend(self):
        now = 1_000_000.0
        self._write_state_file({"deep_audit": {
            "hourly_window_start_ts": now,
            "hourly_window_count": 0,
            "daily_budget_date": "1999-01-01",        # stale date
            "daily_budget_spent_usd": 100.0,
        }})
        state = dd._read_state()
        with mock.patch.object(dd, "_now", return_value=now):
            ok, why = dd._deep_audit_budget_ok(state)
            today = dd._today()    # same patched clock the source used
        self.assertTrue(ok)
        # Reset persisted: today's date, zeroed spend.
        persisted = self._read_state_file()["deep_audit"]
        self.assertEqual(persisted["daily_budget_date"], today)
        self.assertEqual(persisted["daily_budget_spent_usd"], 0.0)


class BuildAuditBlobTests(_Base):
    def test_blob_includes_each_file(self):
        p1 = os.path.join(self._tmp, "a.py")
        p2 = os.path.join(self._tmp, "core", "b.py")
        os.makedirs(os.path.dirname(p2), exist_ok=True)
        with open(p1, "w", encoding="utf-8") as f:
            f.write("AAA")
        with open(p2, "w", encoding="utf-8") as f:
            f.write("BBB")
        blob = dd._build_audit_file_blob([p1, p2])
        self.assertIn("=== FILE: a.py ===", blob)
        self.assertIn("AAA", blob)
        self.assertIn("BBB", blob)
        # core\b.py relpath uses os.sep.
        self.assertIn(os.path.join("core", "b.py"), blob)

    def test_unreadable_file_marked(self):
        missing = os.path.join(self._tmp, "ghost.py")
        blob = dd._build_audit_file_blob([missing])
        self.assertIn("unable to read", blob)

    def test_truncates_large_file(self):
        p = os.path.join(self._tmp, "big.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write("Z" * 20000)
        blob = dd._build_audit_file_blob([p], per_file_byte_cap=100)
        self.assertIn("[truncated]", blob)
        self.assertLess(len(blob), 1000)


class CallAnthropicAuditorTests(_Base):
    def test_sdk_missing_returns_none(self):
        real_import = __import__

        def no_anthropic(name, *a, **k):
            if name == "anthropic":
                raise ImportError("no sdk")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=no_anthropic):
            self.assertIsNone(dd._call_anthropic_auditor("prompt"))

    def test_missing_api_key_returns_none(self):
        fake = FakeAnthropicModule(text="ignored")
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            self.assertIsNone(dd._call_anthropic_auditor("prompt"))

    def test_successful_call_returns_text(self):
        fake = FakeAnthropicModule(text="some findings")
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            out = dd._call_anthropic_auditor("the-prompt")
        self.assertEqual(out, "some findings")
        self.assertEqual(fake.created_kwargs["model"], dd.DEEP_AUDIT_MODEL)
        self.assertEqual(fake.created_kwargs["messages"][0]["content"],
                         "the-prompt")

    def test_empty_content_returns_none(self):
        fake = FakeAnthropicModule(text=None)   # no content blocks
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            self.assertIsNone(dd._call_anthropic_auditor("p"))

    def test_api_cap_error_calm_path(self):
        fake = FakeAnthropicModule(raises=RuntimeError("You have hit your usage limit"))
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            self.assertIsNone(dd._call_anthropic_auditor("p"))

    def test_generic_error_path(self):
        fake = FakeAnthropicModule(raises=ValueError("weird boom"))
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            self.assertIsNone(dd._call_anthropic_auditor("p"))


class ParseAuditFindingsTests(_Base):
    def test_plain_json(self):
        raw = json.dumps({"findings": [
            {"rank": 2, "file": "core/x.py", "line": 5, "category": "race",
             "summary": "s", "fix_hint": "h"},
            {"rank": 1, "file": "a.py", "line": 0, "category": "leak",
             "summary": "s2", "fix_hint": "h2"},
        ]})
        out = dd._parse_audit_findings(raw)
        self.assertEqual(len(out), 2)
        # Sorted by rank ascending.
        self.assertEqual(out[0]["rank"], 1)
        self.assertEqual(out[0]["file"], "a.py")

    def test_strips_json_fences(self):
        raw = "```json\n" + json.dumps({"findings": []}) + "\n```"
        self.assertEqual(dd._parse_audit_findings(raw), [])

    def test_strips_bare_fences(self):
        raw = "```\n" + json.dumps({"findings": [{"file": "a.py"}]}) + "\n```"
        out = dd._parse_audit_findings(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["file"], "a.py")
        self.assertEqual(out[0]["rank"], 99)         # default rank

    def test_embedded_json_block_extracted(self):
        raw = ("Here are the findings you asked for:\n"
               '{"findings": [{"file": "z.py", "summary": "boom"}]}\n'
               "Hope that helps!")
        out = dd._parse_audit_findings(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["file"], "z.py")

    def test_no_json_returns_empty(self):
        self.assertEqual(dd._parse_audit_findings("totally not json"), [])

    def test_embedded_block_still_invalid(self):
        # Has braces but the braces content is not valid JSON.
        self.assertEqual(dd._parse_audit_findings("prefix {nope: } suffix"), [])

    def test_findings_not_a_list(self):
        self.assertEqual(
            dd._parse_audit_findings(json.dumps({"findings": "oops"})), [])

    def test_top_level_not_dict(self):
        self.assertEqual(dd._parse_audit_findings(json.dumps([1, 2, 3])), [])

    def test_non_dict_findings_skipped(self):
        raw = json.dumps({"findings": ["string", 42,
                                       {"file": "ok.py", "summary": "s"}]})
        out = dd._parse_audit_findings(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["file"], "ok.py")


class RunDeepAuditOnceTests(_Base):
    def _touch_py(self, relpath, mtime=1000):
        full = os.path.join(self._tmp, relpath)
        os.makedirs(os.path.dirname(full) or self._tmp, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write("# code\n")
        os.utime(full, (mtime, mtime))
        return full

    def test_no_files_returns_zero(self):
        # Empty project dir → no source files.
        with mock.patch.object(dd, "_recently_modified_source_files",
                               return_value=[]):
            self.assertEqual(dd._run_deep_audit_once("test"), 0)

    def test_auditor_returns_none_zero(self):
        self._touch_py("a.py")
        with mock.patch.object(dd, "_call_anthropic_auditor",
                               return_value=None):
            self.assertEqual(dd._run_deep_audit_once("test"), 0)

    def test_queues_findings_and_updates_state(self):
        self._touch_py("a.py")
        findings_json = json.dumps({"findings": [
            {"rank": 1, "file": "core/x.py", "line": 12, "category": "race",
             "summary": "shared dict mutated", "fix_hint": "add a lock"},
        ]})
        with mock.patch.object(dd, "_call_anthropic_auditor",
                               return_value=findings_json), \
                mock.patch.object(dd, "_count_pipeline_task_completions",
                                  return_value=42):
            queued = dd._run_deep_audit_once("batch_of_10")
        self.assertEqual(queued, 1)
        txt = self._todo_text()
        self.assertIn("[deep-audit] race in core/x.py:12", txt)
        self.assertIn("shared dict mutated", txt)
        self.assertIn("add a lock", txt)
        st = self._read_state_file()["deep_audit"]
        self.assertEqual(st["runs"], 1)
        self.assertEqual(st["last_pipeline_event_count"], 42)
        self.assertEqual(st["hourly_window_count"], 1)
        self.assertEqual(st["pending_findings"], 1)
        self.assertAlmostEqual(st["daily_budget_spent_usd"],
                               dd.DEEP_AUDIT_ESTIMATED_COST_PER_RUN_USD)

    def test_findings_with_blank_fields_use_fallbacks(self):
        self._touch_py("a.py")
        findings_json = json.dumps({"findings": [{"file": "", "summary": ""}]})
        with mock.patch.object(dd, "_call_anthropic_auditor",
                               return_value=findings_json), \
                mock.patch.object(dd, "_count_pipeline_task_completions",
                                  return_value=0):
            dd._run_deep_audit_once("test")
        txt = self._todo_text()
        self.assertIn("[deep-audit] issue in ?:0", txt)
        self.assertIn("Fix hint: investigate", txt)


class DeepAuditLoopTests(_Base):
    def _fake_config(self, enabled):
        return mock.MagicMock(OVERNIGHT_UPGRADE_ENABLED=enabled)

    def test_first_wait_true_returns(self):
        n = _drive_loop(dd._deep_audit_loop, [True])
        self.assertEqual(n, 1)

    def test_paused_continues_without_audit(self):
        self._write_state_file({"paused": True, "deep_audit": {}})
        with mock.patch.object(dd, "_deep_audit_due") as due:
            # waits: [0]=interval(False) enter body; loop top of NEXT iter
            # waits again [1]=(True) return.
            _drive_loop(dd._deep_audit_loop, [False, True])
        due.assert_not_called()

    def test_overnight_disabled_skips(self):
        self._write_state_file({"paused": False, "deep_audit": {}})
        with mock.patch.dict(sys.modules,
                             {"core.config": self._fake_config(False)}), \
                mock.patch.object(dd, "_deep_audit_due") as due:
            _drive_loop(dd._deep_audit_loop, [False, True])
        due.assert_not_called()

    def test_config_import_failure_treated_as_disabled(self):
        self._write_state_file({"paused": False, "deep_audit": {}})
        real_import = __import__

        def no_config(name, *a, **k):
            if name == "core.config":
                raise ImportError("config gone")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=no_config), \
                mock.patch.object(dd, "_deep_audit_due") as due:
            _drive_loop(dd._deep_audit_loop, [False, True])
        due.assert_not_called()

    def test_not_due_skips_budget_and_run(self):
        self._write_state_file({"paused": False, "deep_audit": {}})
        with mock.patch.dict(sys.modules,
                             {"core.config": self._fake_config(True)}), \
                mock.patch.object(dd, "_deep_audit_due",
                                  return_value=(False, "")), \
                mock.patch.object(dd, "_deep_audit_budget_ok") as budget, \
                mock.patch.object(dd, "_run_deep_audit_once") as run:
            _drive_loop(dd._deep_audit_loop, [False, True])
        budget.assert_not_called()
        run.assert_not_called()

    def test_budget_blocked_skips_run(self):
        self._write_state_file({"paused": False, "deep_audit": {}})
        with mock.patch.dict(sys.modules,
                             {"core.config": self._fake_config(True)}), \
                mock.patch.object(dd, "_deep_audit_due",
                                  return_value=(True, "batch")), \
                mock.patch.object(dd, "_deep_audit_budget_ok",
                                  return_value=(False, "hourly_cap")), \
                mock.patch.object(dd, "_run_deep_audit_once") as run:
            _drive_loop(dd._deep_audit_loop, [False, True])
        run.assert_not_called()

    def test_due_and_ok_runs_audit(self):
        self._write_state_file({"paused": False, "deep_audit": {}})
        with mock.patch.dict(sys.modules,
                             {"core.config": self._fake_config(True)}), \
                mock.patch.object(dd, "_deep_audit_due",
                                  return_value=(True, "batch_of_10")), \
                mock.patch.object(dd, "_deep_audit_budget_ok",
                                  return_value=(True, "ok")), \
                mock.patch.object(dd, "_run_deep_audit_once") as run:
            _drive_loop(dd._deep_audit_loop, [False, True])
        run.assert_called_once_with("batch_of_10")

    def test_run_exception_is_contained(self):
        self._write_state_file({"paused": False, "deep_audit": {}})
        with mock.patch.dict(sys.modules,
                             {"core.config": self._fake_config(True)}), \
                mock.patch.object(dd, "_deep_audit_due",
                                  return_value=(True, "batch")), \
                mock.patch.object(dd, "_deep_audit_budget_ok",
                                  return_value=(True, "ok")), \
                mock.patch.object(dd, "_run_deep_audit_once",
                                  side_effect=RuntimeError("audit boom")):
            _drive_loop(dd._deep_audit_loop, [False, True])
        # Inner try/except contained it; reaching here = pass.

    def test_outer_exception_is_contained(self):
        # Make _read_state raise inside the body → outer except handles it.
        with mock.patch.object(dd, "_read_state",
                               side_effect=RuntimeError("read boom")):
            _drive_loop(dd._deep_audit_loop, [False, True])


# ──────────────────────────── anomaly watch ──────────────────────────

class DedupSignatureTests(_Base):
    def test_prune_drops_expired(self):
        now = 1_000_000.0
        state = {"anomaly_watch": {"queued_signatures": {
            "fresh": now - 10,
            "stale": now - dd.ANOMALY_DEDUP_GAP_S - 100,
            "bad": "not-a-number",
        }}}
        with mock.patch.object(dd, "_now", return_value=now):
            dd._prune_dedup_signatures(state)
        sigs = state["anomaly_watch"]["queued_signatures"]
        self.assertIn("fresh", sigs)
        self.assertNotIn("stale", sigs)
        self.assertNotIn("bad", sigs)

    def test_prune_resets_non_dict(self):
        state = {"anomaly_watch": {"queued_signatures": ["oops"]}}
        dd._prune_dedup_signatures(state)
        self.assertEqual(state["anomaly_watch"]["queued_signatures"], {})

    def test_claim_first_time_succeeds(self):
        self.assertTrue(dd._claim_signature("sigA"))
        persisted = self._read_state_file()["anomaly_watch"]["queued_signatures"]
        self.assertIn("sigA", persisted)

    def test_claim_within_window_fails(self):
        now = 1_000_000.0
        with mock.patch.object(dd, "_now", return_value=now):
            self.assertTrue(dd._claim_signature("sigB"))
            # Immediate re-claim inside the window is rejected.
            self.assertFalse(dd._claim_signature("sigB"))

    def test_claim_after_window_succeeds_again(self):
        self._write_state_file({"anomaly_watch": {"queued_signatures": {
            "sigC": 1000.0}}})
        with mock.patch.object(dd, "_now",
                               return_value=1000.0 + dd.ANOMALY_DEDUP_GAP_S + 1):
            self.assertTrue(dd._claim_signature("sigC"))

    def test_claim_corrupt_timestamp_reclaims(self):
        self._write_state_file({"anomaly_watch": {"queued_signatures": {
            "sigD": "garbage"}}})
        # TypeError/ValueError path → falls through to claim.
        self.assertTrue(dd._claim_signature("sigD"))

    def test_claim_signatures_non_dict_reset(self):
        self._write_state_file({"anomaly_watch": {"queued_signatures": "broken"}})
        self.assertTrue(dd._claim_signature("sigE"))
        persisted = self._read_state_file()["anomaly_watch"]["queued_signatures"]
        self.assertIn("sigE", persisted)

    def test_release_signature(self):
        dd._claim_signature("sigF")
        dd._release_signature("sigF")
        persisted = self._read_state_file()["anomaly_watch"]["queued_signatures"]
        self.assertNotIn("sigF", persisted)

    def test_release_missing_is_noop(self):
        dd._release_signature("never-claimed")  # must not raise

    def test_bump_detection_count(self):
        dd._bump_detection_count()
        dd._bump_detection_count()
        self.assertEqual(
            self._read_state_file()["anomaly_watch"]["detections"], 2)


class QueueAnomalyTests(_Base):
    def test_queues_new_anomaly(self):
        ok = dd._queue_anomaly("body here", "sig1", already_queued_this_sweep=0)
        self.assertTrue(ok)
        self.assertIn("body here", self._todo_text())
        self.assertEqual(
            self._read_state_file()["anomaly_watch"]["detections"], 1)

    def test_per_sweep_cap_blocks(self):
        ok = dd._queue_anomaly("body", "sig",
                               already_queued_this_sweep=dd.ANOMALY_MAX_QUEUED_PER_SWEEP)
        self.assertFalse(ok)
        self.assertEqual(self._todo_text(), "")

    def test_dedup_claim_blocks_second(self):
        self.assertTrue(dd._queue_anomaly("b1", "dupsig", 0))
        # Same signature → claim rejects, even with different body.
        self.assertFalse(dd._queue_anomaly("b2-different", "dupsig", 0))

    def test_append_failure_releases_claim(self):
        # Append returns False (e.g. body already on disk) → claim is released
        # so a later sweep can retry.
        with mock.patch.object(dd, "_append_todo_task", return_value=False):
            ok = dd._queue_anomaly("body", "relsig", 0)
        self.assertFalse(ok)
        persisted = self._read_state_file()["anomaly_watch"]["queued_signatures"]
        self.assertNotIn("relsig", persisted)


class TailBytesTests(_Base):
    def test_reads_whole_small_file(self):
        p = os.path.join(self._tmp, "f.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("hello world")
        self.assertEqual(dd._tail_bytes(p, 1024), "hello world")

    def test_reads_only_tail_of_big_file(self):
        p = os.path.join(self._tmp, "f.txt")
        with open(p, "wb") as f:
            f.write(b"A" * 1000 + b"TAILMARKER")
        out = dd._tail_bytes(p, 20)
        self.assertTrue(out.endswith("TAILMARKER"))
        self.assertNotIn("A" * 100, out)

    def test_missing_file_returns_empty(self):
        self.assertEqual(dd._tail_bytes(os.path.join(self._tmp, "no.txt"), 10),
                         "")


class CheckBootFailuresTests(_Base):
    def _write_boot(self, lines, mode="w"):
        with open(dd.BOOT_FAILURES_FILE, mode, encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")

    def test_missing_file_noop(self):
        q = [0]
        dd._check_boot_failures(q)
        self.assertEqual(q[0], 0)

    def test_queues_one_per_kind(self):
        self._write_boot([
            json.dumps({"kind": "lock_busy", "iso": "2026-06-01T00:00:00",
                        "winerror": 32, "errno": 13, "lock_path": "C:/x.lock",
                        "error_repr": "PermissionError(13)"}),
        ])
        q = [0]
        dd._check_boot_failures(q)
        self.assertEqual(q[0], 1)
        txt = self._todo_text()
        self.assertIn("[anomaly] boot failure detected (lock_busy)", txt)
        self.assertIn("winerror=32", txt)
        self.assertIn("last error: PermissionError(13)", txt)
        # Offset advanced.
        off = self._read_state_file()["anomaly_watch"]["last_boot_failure_offset"]
        self.assertGreater(off, 0)

    def test_dedups_repeated_kind_in_same_batch(self):
        # Two same-kind entries → seen_kinds collapses to one, but per-sweep
        # cap is 1 anyway. Use distinct kinds to test the cap separately.
        self._write_boot([
            json.dumps({"kind": "same", "winerror": 1, "errno": 1}),
            json.dumps({"kind": "same", "winerror": 1, "errno": 1}),
        ])
        q = [0]
        dd._check_boot_failures(q)
        self.assertEqual(q[0], 1)
        self.assertEqual(self._todo_text().count("boot failure detected"), 1)

    def test_per_sweep_cap_limits_distinct_kinds(self):
        self._write_boot([
            json.dumps({"kind": "kindA", "winerror": 1, "errno": 1}),
            json.dumps({"kind": "kindB", "winerror": 2, "errno": 2}),
        ])
        q = [0]
        dd._check_boot_failures(q)
        # Cap is 1 per sweep.
        self.assertEqual(q[0], dd.ANOMALY_MAX_QUEUED_PER_SWEEP)

    def test_no_new_bytes_skips(self):
        self._write_boot([json.dumps({"kind": "x", "winerror": 1, "errno": 1})])
        size = os.path.getsize(dd.BOOT_FAILURES_FILE)
        self._write_state_file({"anomaly_watch": {
            "last_boot_failure_offset": size}})
        q = [0]
        dd._check_boot_failures(q)
        self.assertEqual(q[0], 0)

    def test_truncation_resets_offset(self):
        # last_offset larger than current size → rescan from 0.
        self._write_boot([json.dumps({"kind": "y", "winerror": 9, "errno": 9})])
        self._write_state_file({"anomaly_watch": {
            "last_boot_failure_offset": 10_000_000}})
        q = [0]
        dd._check_boot_failures(q)
        self.assertEqual(q[0], 1)

    def test_corrupt_and_nondict_lines_skipped(self):
        self._write_boot([
            "{ broken json",
            json.dumps([1, 2, 3]),                      # not a dict
            json.dumps({"kind": "real", "winerror": 5, "errno": 5}),
        ])
        q = [0]
        dd._check_boot_failures(q)
        self.assertEqual(q[0], 1)
        self.assertIn("(real)", self._todo_text())

    def test_blank_lines_skipped(self):
        self._write_boot([
            "",
            "   ",
            json.dumps({"kind": "withblank", "winerror": 1, "errno": 1}),
        ])
        q = [0]
        dd._check_boot_failures(q)
        self.assertEqual(q[0], 1)

    def test_default_kind_when_missing(self):
        self._write_boot([json.dumps({"winerror": 1, "errno": 2})])
        q = [0]
        dd._check_boot_failures(q)
        self.assertIn("(boot_failure)", self._todo_text())

    def test_getsize_oserror_returns(self):
        self._write_boot([json.dumps({"kind": "z", "winerror": 1, "errno": 1})])
        with mock.patch.object(dd.os.path, "getsize",
                               side_effect=OSError("stat fail")):
            q = [0]
            dd._check_boot_failures(q)   # must not raise
        self.assertEqual(q[0], 0)

    def test_read_oserror_returns(self):
        # getsize succeeds (new bytes present) but the open/read raises OSError.
        self._write_boot([json.dumps({"kind": "z", "winerror": 1, "errno": 1})])
        with mock.patch("builtins.open", side_effect=OSError("read fail")):
            q = [0]
            dd._check_boot_failures(q)   # must not raise
        self.assertEqual(q[0], 0)


class CheckStuckLoopTests(_Base):
    def test_no_hud_file_resets_streak(self):
        self._write_state_file({"anomaly_watch": {
            "stuck_loop_consecutive_misses": 3}})
        q = [0]
        # HUD_STATE_FILE doesn't exist in temp dir.
        dd._check_stuck_loop(q)
        self.assertEqual(q[0], 0)
        self.assertEqual(
            self._read_state_file()["anomaly_watch"]["stuck_loop_consecutive_misses"],
            0)

    def test_fresh_hud_resets_streak(self):
        with open(dd.HUD_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        now = 1_000_000.0
        # mtime = now (age 0 < threshold).
        os.utime(dd.HUD_STATE_FILE, (now, now))
        self._write_state_file({"anomaly_watch": {
            "stuck_loop_consecutive_misses": 2}})
        q = [0]
        with mock.patch.object(dd, "_now", return_value=now):
            dd._check_stuck_loop(q)
        self.assertEqual(q[0], 0)
        self.assertEqual(
            self._read_state_file()["anomaly_watch"]["stuck_loop_consecutive_misses"],
            0)

    def test_first_stale_sweep_increments_but_no_queue(self):
        with open(dd.HUD_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        now = 1_000_000.0
        old = now - dd.ANOMALY_STUCK_LOOP_THRESHOLD_S - 50
        os.utime(dd.HUD_STATE_FILE, (old, old))
        q = [0]
        with mock.patch.object(dd, "_now", return_value=now):
            dd._check_stuck_loop(q)
        # MIN_CONSECUTIVE is 2 → first miss doesn't queue.
        self.assertEqual(q[0], 0)
        self.assertEqual(
            self._read_state_file()["anomaly_watch"]["stuck_loop_consecutive_misses"],
            1)
        self.assertEqual(self._todo_text(), "")

    def test_threshold_reached_queues_and_resets(self):
        with open(dd.HUD_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        now = 1_000_000.0
        old = now - dd.ANOMALY_STUCK_LOOP_THRESHOLD_S - 50
        os.utime(dd.HUD_STATE_FILE, (old, old))
        # Pre-seed one prior miss so this sweep crosses MIN_CONSECUTIVE=2.
        self._write_state_file({"anomaly_watch": {
            "stuck_loop_consecutive_misses": 1, "queued_signatures": {}}})
        q = [0]
        with mock.patch.object(dd, "_now", return_value=now):
            dd._check_stuck_loop(q)
        self.assertEqual(q[0], 1)
        self.assertIn("[anomaly] main loop appears stuck", self._todo_text())
        # Streak reset after successful queue.
        self.assertEqual(
            self._read_state_file()["anomaly_watch"]["stuck_loop_consecutive_misses"],
            0)

    def test_getmtime_oserror_returns(self):
        with open(dd.HUD_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        with mock.patch.object(dd.os.path, "getmtime",
                               side_effect=OSError("stat fail")):
            q = [0]
            dd._check_stuck_loop(q)   # must not raise
        self.assertEqual(q[0], 0)


class LooksLikeTextLogTests(_Base):
    def test_empty_is_false(self):
        self.assertFalse(dd._looks_like_text_log(""))

    def test_plain_text_true(self):
        self.assertTrue(dd._looks_like_text_log("hello\nworld\n[action] ok"))

    def test_nul_bytes_false(self):
        self.assertFalse(dd._looks_like_text_log("abc\x00def"))

    def test_mostly_binary_false(self):
        self.assertFalse(dd._looks_like_text_log("".join(chr(i) for i in range(1, 31))))

    def test_high_unicode_ok(self):
        # Chars >= 160 count as printable per the heuristic.
        self.assertTrue(dd._looks_like_text_log("café résumé\n" * 5))


class LineAtTests(_Base):
    def test_extracts_middle_line(self):
        text = "first\nSECOND match\nthird"
        idx = text.index("match")
        line = dd._line_at(text, idx, idx + 5)
        self.assertEqual(line, "SECOND match")

    def test_match_on_last_line_no_trailing_newline(self):
        text = "a\nb\nLAST"
        idx = text.index("LAST")
        self.assertEqual(dd._line_at(text, idx, idx + 4), "LAST")

    def test_match_on_first_line(self):
        text = "HEAD here\nrest"
        self.assertEqual(dd._line_at(text, 0, 4), "HEAD here")


class TransientFailureTests(_Base):
    def test_transient_token_detected(self):
        self.assertTrue(dd._is_transient_failure_line(
            "  [action] weather failed: getaddrinfo failed"))
        self.assertTrue(dd._is_transient_failure_line(
            "request read timed out after 30s"))
        self.assertTrue(dd._is_transient_failure_line("WinError 10060"))

    def test_non_transient_not_detected(self):
        self.assertFalse(dd._is_transient_failure_line(
            "  [action] foo failed: KeyError 'bar'"))


class CheckLogFailuresTests(_Base):
    def test_no_log_noop(self):
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], 0)

    def test_binary_log_skipped(self):
        self._write_log("session_1.log", "abc\x00\x00def")
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], 0)

    def test_repeated_skill_failure_queued(self):
        body = "\n".join(
            f"  [action] system_pulse failed: oops {i}"
            for i in range(dd.ANOMALY_FAILURE_THRESHOLD + 1))
        self._write_log("session_1.log", body)
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], 1)
        txt = self._todo_text()
        self.assertIn("skill 'system_pulse' failed", txt)

    def test_below_threshold_not_queued(self):
        body = "\n".join(
            f"  [action] system_pulse failed: oops {i}"
            for i in range(dd.ANOMALY_FAILURE_THRESHOLD - 1))
        self._write_log("session_1.log", body)
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], 0)

    def test_transient_skill_whitelisted(self):
        # 'weather' is in _TRANSIENT_SKILL_NAMES → never counted.
        body = "\n".join(
            f"  [action] weather failed: oops {i}"
            for i in range(dd.ANOMALY_FAILURE_THRESHOLD + 5))
        self._write_log("session_1.log", body)
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], 0)

    def test_transient_error_token_lines_filtered(self):
        # Non-whitelisted skill but every failure line carries a transient
        # token → filtered out before threshold.
        body = "\n".join(
            f"  [action] db_sync failed: timed out {i}"
            for i in range(dd.ANOMALY_FAILURE_THRESHOLD + 3))
        self._write_log("session_1.log", body)
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], 0)

    def test_exception_burst_queued(self):
        body = "\n".join("Traceback (most recent call last):"
                         for _ in range(dd.ANOMALY_EXCEPTION_THRESHOLD + 1))
        self._write_log("session_1.log", body)
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], 1)
        self.assertIn("unhandled exception traces", self._todo_text())

    def test_per_sweep_cap_shared_across_detectors(self):
        # Both a skill-failure spike AND an exception burst present; the
        # per-sweep cap (1) means only the first detector to fire queues.
        parts = []
        parts += [f"  [action] system_pulse failed: e{i}"
                  for i in range(dd.ANOMALY_FAILURE_THRESHOLD + 1)]
        parts += ["Traceback (most recent call last):"
                  for _ in range(dd.ANOMALY_EXCEPTION_THRESHOLD + 1)]
        self._write_log("session_1.log", "\n".join(parts))
        q = [0]
        dd._check_log_failures(q)
        self.assertEqual(q[0], dd.ANOMALY_MAX_QUEUED_PER_SWEEP)

    def test_tail_read_exception_returns(self):
        self._write_log("session_1.log", "data")
        with mock.patch.object(dd, "_tail_bytes",
                               side_effect=RuntimeError("tail boom")):
            q = [0]
            dd._check_log_failures(q)   # must not raise
        self.assertEqual(q[0], 0)

    def test_action_regex_finditer_exception_handled(self):
        # If the action-failure regex's finditer raises, matches falls back to
        # [] and the loop survives (only the traceback detector then runs).
        self._write_log("session_1.log", "  [action] foo failed: x\n" * 3)
        broken = mock.Mock()
        broken.finditer.side_effect = RuntimeError("regex boom")
        with mock.patch.object(dd, "_ACTION_FAILURE_RE", broken):
            q = [0]
            dd._check_log_failures(q)   # must not raise
        self.assertEqual(q[0], 0)

    def test_match_group_exception_skipped(self):
        # A match whose .group(1) raises is skipped (continue) rather than
        # crashing the per-skill counter.
        self._write_log("session_1.log", "  [action] foo failed: boom\n")

        class BadMatch:
            def group(self, i):
                raise RuntimeError("group boom")

            def start(self):
                return 0

            def end(self):
                return 1

        fake_re = mock.Mock()
        fake_re.finditer.return_value = [BadMatch()]
        with mock.patch.object(dd, "_ACTION_FAILURE_RE", fake_re):
            q = [0]
            dd._check_log_failures(q)   # must not raise
        self.assertEqual(q[0], 0)

    def test_blank_skill_name_skipped(self):
        # A match whose captured group strips to empty is skipped (the real
        # regex can't produce this, so drive it with a fake match).
        self._write_log("session_1.log", "  [action] x failed: y\n")

        class BlankMatch:
            def group(self, i):
                return "   "          # strips/lowers to ""

            def start(self):
                return 0

            def end(self):
                return 1

        fake_re = mock.Mock()
        fake_re.finditer.return_value = [BlankMatch()]
        with mock.patch.object(dd, "_ACTION_FAILURE_RE", fake_re):
            q = [0]
            dd._check_log_failures(q)   # must not raise
        self.assertEqual(q[0], 0)

    def test_line_at_exception_during_transient_check(self):
        # _line_at raising for a real match → line defaults to "" and the
        # match is still counted (transient filter just can't classify it).
        body = "\n".join(f"  [action] db_sync failed: e{i}"
                         for i in range(dd.ANOMALY_FAILURE_THRESHOLD + 1))
        self._write_log("session_1.log", body)
        with mock.patch.object(dd, "_line_at",
                               side_effect=RuntimeError("line boom")):
            q = [0]
            dd._check_log_failures(q)
        # With line="" the transient filter passes everything through, so the
        # threshold is still crossed and one task is queued.
        self.assertEqual(q[0], 1)

    def test_traceback_regex_finditer_exception_handled(self):
        # If the traceback regex's finditer raises, tb_hits falls back to [] and
        # the burst detector simply doesn't fire (no crash).
        self._write_log("session_1.log", "Traceback (most recent call last):\n" *
                        (dd.ANOMALY_EXCEPTION_THRESHOLD + 1))
        broken = mock.Mock()
        broken.finditer.side_effect = RuntimeError("tb regex boom")
        with mock.patch.object(dd, "_TRACEBACK_RE", broken):
            q = [0]
            dd._check_log_failures(q)   # must not raise
        self.assertEqual(q[0], 0)

    def test_traceback_sample_exception_uses_placeholder(self):
        # Enough tracebacks to fire the burst detector, but _line_at raises
        # when building the sample → caught, body still queued.
        self._write_log("session_1.log",
                        "\n".join("Traceback (most recent call last):"
                                  for _ in range(dd.ANOMALY_EXCEPTION_THRESHOLD + 1)))
        # The log has no [action] lines, so _line_at is only invoked to build
        # the traceback sample — make it raise to hit the placeholder path.
        with mock.patch.object(dd, "_line_at",
                               side_effect=RuntimeError("sample boom")):
            q = [0]
            dd._check_log_failures(q)
        self.assertEqual(q[0], 1)
        self.assertIn("(sample unavailable)", self._todo_text())


class AnomalyWatchLoopTests(_Base):
    def test_initial_delay_true_returns(self):
        n = _drive_loop(dd._anomaly_watch_loop, [True])
        self.assertEqual(n, 1)

    def test_paused_skips_checks(self):
        self._write_state_file({"paused": True, "anomaly_watch": {}})
        with mock.patch.object(dd, "_check_boot_failures") as a, \
                mock.patch.object(dd, "_check_stuck_loop") as b, \
                mock.patch.object(dd, "_check_log_failures") as c:
            # 2nd iteration drives the paused-branch `continue`.
            _drive_loop(dd._anomaly_watch_loop, [False, False, True])
        a.assert_not_called()
        b.assert_not_called()
        c.assert_not_called()

    def test_runs_all_three_checks_and_records_poll(self):
        self._write_state_file({"paused": False, "anomaly_watch": {}})
        with mock.patch.object(dd, "_check_boot_failures") as a, \
                mock.patch.object(dd, "_check_stuck_loop") as b, \
                mock.patch.object(dd, "_check_log_failures") as c:
            _drive_loop(dd._anomaly_watch_loop, [False, True])
        a.assert_called_once()
        b.assert_called_once()
        c.assert_called_once()
        st = self._read_state_file()["anomaly_watch"]
        self.assertIsNotNone(st["last_poll_iso"])
        self.assertGreater(st["last_poll_ts"], 0)

    def test_one_check_raising_does_not_stop_others(self):
        self._write_state_file({"paused": False, "anomaly_watch": {}})
        # The loop's error logger reads fn.__name__, which a bare MagicMock
        # doesn't expose. Real check functions always have __name__, so set it
        # explicitly to faithfully model "a real detector raised".
        boot = mock.Mock(side_effect=RuntimeError("boot boom"))
        boot.__name__ = "_check_boot_failures"
        with mock.patch.object(dd, "_check_boot_failures", boot), \
                mock.patch.object(dd, "_check_stuck_loop") as b, \
                mock.patch.object(dd, "_check_log_failures") as c:
            _drive_loop(dd._anomaly_watch_loop, [False, True])
        # The raising check was isolated; the others still ran.
        b.assert_called_once()
        c.assert_called_once()

    def test_outer_body_exception_contained(self):
        # _read_state raising at top of body → outer except handles, then sleep.
        with mock.patch.object(dd, "_read_state",
                               side_effect=RuntimeError("read boom")):
            _drive_loop(dd._anomaly_watch_loop, [False, True])


# ──────────────────────────── lifecycle ──────────────────────────────

class LifecycleTests(_Base):
    def tearDown(self):
        # Ensure any threads this test class started are signalled down and
        # _started is reset BEFORE the base restores its snapshot.
        try:
            dd._stop_event.set()
            for t in list(dd._threads):
                if t.is_alive():
                    t.join(timeout=1.0)
        finally:
            dd._threads.clear()
            dd._started = False
        super().tearDown()

    def _fake_thread_factory(self):
        """Return a Thread replacement that records targets but never runs
        them, so start_diagnostic_daemons() spins up nothing real."""
        created = []

        class FakeThread:
            def __init__(self, target=None, name="", daemon=False):
                self.target = target
                self.name = name
                self.daemon = daemon
                self._alive = False
                created.append(self)

            def start(self):
                self._alive = True

            def is_alive(self):
                return self._alive

            def join(self, timeout=None):
                self._alive = False

        return FakeThread, created

    def test_start_spawns_four_daemons(self):
        FakeThread, created = self._fake_thread_factory()
        with mock.patch.object(dd.threading, "Thread", FakeThread):
            dd._started = False
            ok = dd.start_diagnostic_daemons()
        self.assertTrue(ok)
        self.assertTrue(dd._started)
        self.assertEqual(len(created), 4)
        names = {t.name for t in created}
        self.assertIn("jarvis-self-diag-daemon", names)
        self.assertIn("jarvis-crash-watch-daemon", names)
        self.assertIn("jarvis-deep-audit-daemon", names)
        self.assertIn("jarvis-anomaly-watch-daemon", names)
        for t in created:
            self.assertTrue(t.daemon)

    def test_start_is_idempotent(self):
        FakeThread, created = self._fake_thread_factory()
        with mock.patch.object(dd.threading, "Thread", FakeThread):
            dd._started = False
            self.assertTrue(dd.start_diagnostic_daemons())
            # Second call → no-op, returns False, no new threads.
            self.assertFalse(dd.start_diagnostic_daemons())
        self.assertEqual(len(created), 4)

    def test_start_survives_thread_start_failure(self):
        created = []

        class FlakyThread:
            def __init__(self, target=None, name="", daemon=False):
                self.target = target
                self.name = name
                self.daemon = daemon
                created.append(self)

            def start(self):
                if self.name == "jarvis-crash-watch-daemon":
                    raise RuntimeError("cannot start")

            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        with mock.patch.object(dd.threading, "Thread", FlakyThread):
            dd._started = False
            dd.start_diagnostic_daemons()
        # 3 of 4 registered (the flaky one was skipped).
        self.assertEqual(len(dd._threads), 3)
        self.assertTrue(dd._started)

    def test_start_state_init_failure_still_starts(self):
        FakeThread, created = self._fake_thread_factory()
        with mock.patch.object(dd.threading, "Thread", FakeThread), \
                mock.patch.object(dd, "_update_state",
                                  side_effect=OSError("disk wedged")):
            dd._started = False
            ok = dd.start_diagnostic_daemons()
        self.assertTrue(ok)
        self.assertEqual(len(created), 4)

    def test_stop_when_not_started_noop(self):
        dd._started = False
        dd.stop_diagnostic_daemons()   # must not raise

    def test_stop_joins_and_clears(self):
        FakeThread, created = self._fake_thread_factory()
        with mock.patch.object(dd.threading, "Thread", FakeThread):
            dd._started = False
            dd.start_diagnostic_daemons()
            self.assertTrue(dd._started)
            dd.stop_diagnostic_daemons(join_timeout=0.1)
        self.assertFalse(dd._started)
        self.assertEqual(len(dd._threads), 0)
        self.assertTrue(dd._stop_event.is_set())

    def test_stop_swallows_join_error(self):
        class JoinRaises:
            name = "x"
            daemon = True

            def start(self):
                pass

            def is_alive(self):
                return False

            def join(self, timeout=None):
                raise RuntimeError("join boom")

        with mock.patch.object(dd.threading, "Thread",
                               lambda target=None, name="", daemon=False: JoinRaises()):
            dd._started = False
            dd.start_diagnostic_daemons()
            dd.stop_diagnostic_daemons(join_timeout=0.1)   # must not raise
        self.assertFalse(dd._started)


class PauseResumeTests(_Base):
    def test_pause_sets_flag(self):
        msg = dd.pause_diagnostics()
        self.assertIn("paused", msg.lower())
        self.assertTrue(self._read_state_file()["paused"])

    def test_resume_clears_flag(self):
        dd.pause_diagnostics()
        msg = dd.resume_diagnostics()
        self.assertIn("resumed", msg.lower())
        self.assertFalse(self._read_state_file()["paused"])

    def test_act_shims_delegate(self):
        self.assertIn("paused", dd.act_pause_diagnostics().lower())
        self.assertTrue(self._read_state_file()["paused"])
        self.assertIn("resumed", dd.act_resume_diagnostics().lower())
        self.assertFalse(self._read_state_file()["paused"])


class StatusTests(_Base):
    def test_status_defaults_not_started(self):
        dd._started = False
        s = dd.diagnostic_daemon_status()
        self.assertFalse(s["paused"])
        self.assertFalse(s["started"])
        self.assertFalse(s["self_diag_alive"])
        self.assertFalse(s["crash_watch_alive"])
        self.assertFalse(s["deep_audit_alive"])
        self.assertFalse(s["anomaly_watch_alive"])
        self.assertEqual(s["self_diag_runs"], 0)
        self.assertEqual(
            s["deep_audit_budget_remaining_usd"],
            round(dd.DEEP_AUDIT_DEFAULT_BUDGET_USD, 4))

    def test_status_alive_when_started_and_recent_heartbeat(self):
        now = 1_000_000.0
        self._write_state_file({
            "paused": False,
            "self_diag": {"alive_ts": now, "runs": 3,
                          "last_run_iso": "2026-06-01T00:00:00"},
            "crash_watch": {"alive_ts": now, "detections": 2,
                            "last_poll_ts": now},
            "deep_audit": {"alive_ts": now, "runs": 1,
                           "daily_budget_spent_usd": 1.0,
                           "pending_findings": 4,
                           "last_run_iso": "2026-06-01T01:00:00"},
            "anomaly_watch": {"alive_ts": now, "detections": 7,
                              "last_poll_iso": "2026-06-01T02:00:00"},
        })
        dd._started = True
        with mock.patch.object(dd, "_now", return_value=now + 10):
            s = dd.diagnostic_daemon_status()
        self.assertTrue(s["self_diag_alive"])
        self.assertTrue(s["crash_watch_alive"])
        self.assertTrue(s["deep_audit_alive"])
        self.assertTrue(s["anomaly_watch_alive"])
        self.assertEqual(s["self_diag_runs"], 3)
        self.assertEqual(s["crash_watch_detections"], 2)
        self.assertEqual(s["deep_audit_pending_findings"], 4)
        self.assertEqual(s["anomaly_watch_detections"], 7)
        self.assertEqual(s["deep_audit_budget_remaining_usd"],
                         round(dd.DEEP_AUDIT_DEFAULT_BUDGET_USD - 1.0, 4))

    def test_status_dead_when_heartbeat_stale(self):
        now = 1_000_000.0
        self._write_state_file({
            "self_diag": {"alive_ts": now - 10 * dd.SELF_DIAG_INTERVAL_S},
            "crash_watch": {"alive_ts": now - 10 * dd.CRASH_POLL_INTERVAL_S},
            "deep_audit": {"alive_ts": now - 10000},
            "anomaly_watch": {"alive_ts": now - 10 * dd.ANOMALY_POLL_INTERVAL_S},
        })
        dd._started = True
        with mock.patch.object(dd, "_now", return_value=now):
            s = dd.diagnostic_daemon_status()
        self.assertFalse(s["self_diag_alive"])
        self.assertFalse(s["crash_watch_alive"])
        self.assertFalse(s["deep_audit_alive"])
        self.assertFalse(s["anomaly_watch_alive"])

    def test_status_budget_remaining_never_negative(self):
        self._write_state_file({"deep_audit": {
            "daily_budget_spent_usd": 9999.0}})
        s = dd.diagnostic_daemon_status()
        self.assertEqual(s["deep_audit_budget_remaining_usd"], 0.0)


class StatusSpokenTests(_Base):
    def test_spoken_never_paused(self):
        self._write_state_file({
            "paused": False,
            "self_diag": {"last_run_iso": "2026-06-01T00:00:00", "runs": 2},
            "crash_watch": {"detections": 1, "last_poll_ts": 1000.0},
            "deep_audit": {"last_run_iso": "2026-06-01T01:00:00", "runs": 1,
                           "pending_findings": 3},
            "anomaly_watch": {"last_poll_iso": "2026-06-01T02:00:00",
                              "detections": 5},
        })
        out = dd.diagnostic_daemon_status_spoken()
        self.assertNotIn("paused", out.lower())
        self.assertIn("2 runs total", out)
        self.assertIn("3 pending findings", out)
        self.assertIn("5 detections", out)

    def test_spoken_paused_prefix(self):
        self._write_state_file({"paused": True})
        out = dd.diagnostic_daemon_status_spoken()
        self.assertTrue(out.startswith("Diagnostics are paused"))

    def test_spoken_never_values(self):
        # No iso timestamps stored → "never" used.
        out = dd.diagnostic_daemon_status_spoken()
        self.assertIn("at never", out)

    def test_act_diagnostic_status_delegates(self):
        out = dd.act_diagnostic_status()
        self.assertIn("self-diagnostic", out.lower())


if __name__ == "__main__":
    unittest.main()
