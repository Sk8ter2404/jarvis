"""Tests for core.atomic_io — the shared atomic JSON writer that every
state-file writer in JARVIS relies on. If this regresses, concurrent readers
(HUD, overlays, skills) can see half-written JSON, so it's load-bearing."""
import json
import os
import tempfile
import unittest
from unittest import mock

import core.atomic_io as atomic_io
from core.atomic_io import _atomic_write_json, _replace_with_retry


class AtomicWriteJsonTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _roundtrip(self, data, **kw):
        path = os.path.join(self.dir, "state.json")
        _atomic_write_json(path, data, **kw)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_dict_roundtrip(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        self.assertEqual(self._roundtrip(data), data)

    def test_overwrite_is_clean(self):
        path = os.path.join(self.dir, "state.json")
        _atomic_write_json(path, {"v": 1})
        _atomic_write_json(path, {"v": 2})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"v": 2})

    def test_non_ascii_roundtrip(self):
        data = {"greeting": "café — naïve — 日本語", "emoji": "🤖"}
        self.assertEqual(self._roundtrip(data), data)

    def test_indent_none_is_compact(self):
        path = os.path.join(self.dir, "compact.json")
        _atomic_write_json(path, {"a": 1, "b": 2}, indent=None)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("\n", text.strip())

    def test_no_tempfiles_left_behind(self):
        path = os.path.join(self.dir, "state.json")
        _atomic_write_json(path, {"v": 1})
        leftovers = [f for f in os.listdir(self.dir) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [], f"stray tempfiles: {leftovers}")

    def test_list_payload(self):
        data = [{"text": "a"}, {"text": "b"}]
        self.assertEqual(self._roundtrip(data), data)


    def test_default_path_dirname_empty(self):
        # A bare filename has no dirname → the helper falls back to "." so
        # mkstemp lands in cwd. Run from inside the temp dir so we don't litter
        # the repo, and assert the file is created and round-trips.
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            _atomic_write_json("bare.json", {"ok": 1})
            with open(os.path.join(self.dir, "bare.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"ok": 1})
        finally:
            os.chdir(cwd)

    def test_fsync_oserror_is_swallowed(self):
        # fsync isn't available on every filesystem; an OSError from it must NOT
        # fail the write (the os.replace is still atomic). Patch os.fsync to
        # raise and confirm the data still lands.
        path = os.path.join(self.dir, "state.json")
        with mock.patch.object(atomic_io.os, "fsync",
                               side_effect=OSError("no fsync on this fs")):
            _atomic_write_json(path, {"durable": False})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"durable": False})
        # And no tempfile should be left behind.
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])

    def test_replace_failure_unlinks_tempfile_and_raises(self):
        # If the final replace fails, the partial tempfile must be cleaned up
        # and the exception propagated so the caller can log / re-queue.
        path = os.path.join(self.dir, "state.json")
        with mock.patch.object(atomic_io, "_replace_with_retry",
                               side_effect=RuntimeError("replace blew up")):
            with self.assertRaises(RuntimeError):
                _atomic_write_json(path, {"v": 1})
        # No tempfile left behind, and the destination was never created.
        self.assertEqual(os.listdir(self.dir), [])

    def test_replace_failure_unlink_also_failing_still_raises(self):
        # Belt-and-suspenders: even if the cleanup os.unlink itself raises, the
        # ORIGINAL replace exception must still propagate (the inner unlink
        # except is best-effort).
        path = os.path.join(self.dir, "state.json")
        with mock.patch.object(atomic_io, "_replace_with_retry",
                               side_effect=RuntimeError("replace blew up")), \
             mock.patch.object(atomic_io.os, "unlink",
                               side_effect=OSError("unlink denied")):
            with self.assertRaises(RuntimeError):
                _atomic_write_json(path, {"v": 1})


class ReplaceWithRetryTests(unittest.TestCase):
    def test_basic_replace(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.txt")
            dst = os.path.join(d, "dst.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("payload")
            _replace_with_retry(src, dst)
            self.assertFalse(os.path.exists(src))
            with open(dst, encoding="utf-8") as f:
                self.assertEqual(f.read(), "payload")

    def test_posix_branch_calls_replace_directly(self):
        # On a non-"nt" os.name the helper is a single os.replace with no retry
        # loop. Force os.name == "posix" and assert os.replace runs exactly once.
        with mock.patch.object(atomic_io.os, "name", "posix"), \
             mock.patch.object(atomic_io.os, "replace") as rep:
            _replace_with_retry("src", "dst")
        rep.assert_called_once_with("src", "dst")

    def test_windows_retries_on_permission_error_then_succeeds(self):
        # First os.replace raises PermissionError (reader holds dst open), the
        # retry succeeds. time.sleep is stubbed so the backoff doesn't wait.
        attempts = {"n": 0}

        def _replace(src, dst):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise PermissionError("WinError 5")
            return None

        with mock.patch.object(atomic_io.os, "name", "nt"), \
             mock.patch.object(atomic_io.os, "replace", side_effect=_replace), \
             mock.patch.object(atomic_io.time, "sleep") as slept:
            _replace_with_retry("src", "dst")
        self.assertEqual(attempts["n"], 2)        # one failure + one success
        slept.assert_called_once()                # backed off exactly once

    def test_windows_all_retries_exhausted_final_attempt_raises(self):
        # Every attempt (the retry loop AND the final un-caught os.replace)
        # raises PermissionError → the exception propagates so the caller logs.
        with mock.patch.object(atomic_io.os, "name", "nt"), \
             mock.patch.object(atomic_io.os, "replace",
                               side_effect=PermissionError("WinError 5")), \
             mock.patch.object(atomic_io.time, "sleep") as slept:
            with self.assertRaises(PermissionError):
                _replace_with_retry("src", "dst")
        # One sleep per entry in the retry-delay tuple (final bare attempt
        # doesn't sleep).
        self.assertEqual(slept.call_count, len(atomic_io._REPLACE_RETRY_DELAYS_S))


if __name__ == "__main__":
    unittest.main()
