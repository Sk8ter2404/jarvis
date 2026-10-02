"""Audit A89: an unreadable memory/pending_promises.json must never be
overwritten without a copy kept aside.

core.memory used to mark the promises "loaded" BEFORE reading the file. When
the read or the JSON decode failed it printed one line and carried on with an
EMPTY registry, and the very next make_promise / cancel / tick saved that empty
list over the file -- every promise that was pending in it was gone, with no
backup. Now a file that cannot be loaded is copied to
``pending_promises.json.corrupt-<stamp>.bak`` first; if even the copy fails,
saves are refused (the file is left exactly as it was) and the next call
retries the read.

Every test points the module at a per-test temp dir, as tests/test_memory.py
does, so nothing touches the real memory/ folder. stdlib unittest only.
"""
from __future__ import annotations

import builtins
import glob
import json
import os
import tempfile
import unittest
from unittest import mock

from core import memory


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mem_dir = self._tmp.name
        self.promises_file = os.path.join(self.mem_dir, "pending_promises.json")
        for p in (mock.patch.object(memory, "_MEM_DIR", self.mem_dir),
                  mock.patch.object(memory, "_PROMISES_FILE", self.promises_file)):
            p.start()
            self.addCleanup(p.stop)
        self._reset()
        self.addCleanup(self._reset)
        self.addCleanup(memory.stop_watcher)
        memory._announce_fn[0] = lambda message, source="promise": None

    def _reset(self):
        with memory._lock:
            memory._promises[:] = []
            memory._next_id[0] = 1
            memory._loaded[0] = False
            if hasattr(memory, "_load_blocked"):
                memory._load_blocked[0] = False
        memory._announce_fn[0] = None

    def _write_raw(self, text: str) -> None:
        with open(self.promises_file, "w", encoding="utf-8") as f:
            f.write(text)

    def _read_raw(self) -> str:
        with open(self.promises_file, "r", encoding="utf-8") as f:
            return f.read()

    def _backups(self):
        return sorted(glob.glob(self.promises_file + ".corrupt-*.bak"))


class CorruptFileIsKeptTests(_Base):
    def test_corrupt_json_is_copied_aside_before_the_next_save(self):
        garbage = '[{"id": 1, "message": "feed the cat", "condition": "manual"'
        self._write_raw(garbage)
        memory.make_promise("water the plants", "manual", source="t")
        baks = self._backups()
        self.assertEqual(len(baks), 1, "no backup kept of the unreadable file")
        with open(baks[0], "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), garbage)
        # The new promise is still saved normally.
        with open(self.promises_file, "r", encoding="utf-8") as f:
            self.assertEqual([p["message"] for p in json.load(f)],
                             ["water the plants"])

    def test_non_list_json_is_copied_aside_too(self):
        payload = json.dumps({"promises": [{"id": 3, "message": "call back"}]})
        self._write_raw(payload)
        memory.make_promise("something new", "manual", source="t")
        baks = self._backups()
        self.assertEqual(len(baks), 1)
        with open(baks[0], "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), payload)

    def test_a_readable_file_makes_no_backup(self):
        memory.make_promise("one", "manual", source="t")
        self._reset()
        memory.make_promise("two", "manual", source="t")
        self.assertEqual(self._backups(), [])
        self.assertEqual(
            sorted(p["message"] for p in memory.list_promises()), ["one", "two"])

    def test_a_missing_file_makes_no_backup(self):
        memory.make_promise("first ever", "manual", source="t")
        self.assertEqual(self._backups(), [])


class UnreadableAndUncopyableTests(_Base):
    """The read fails AND the copy fails (a locked file): nothing may be
    written over it, and the next call retries the read."""

    def _locked(self):
        real_open = builtins.open
        target = os.path.normcase(os.path.abspath(self.promises_file))

        def fake_open(file, mode="r", *args, **kwargs):
            if (isinstance(file, str) and "r" in mode
                    and os.path.normcase(os.path.abspath(file)) == target):
                raise PermissionError(13, "file is locked", file)
            return real_open(file, mode, *args, **kwargs)

        return (mock.patch("builtins.open", fake_open),
                mock.patch("shutil.copy2",
                           side_effect=PermissionError(13, "locked")))

    def _seed_two_pending(self):
        memory.make_promise("pending one", "manual", source="t")
        memory.make_promise("pending two", "manual", source="t")
        self._reset()
        return self._read_raw()

    def test_locked_file_is_not_overwritten(self):
        original = self._seed_two_pending()
        p_open, p_copy = self._locked()
        with p_open, p_copy:
            memory.make_promise("made while locked", "manual", source="t")
        self.assertEqual(self._read_raw(), original,
                         "the unreadable file was overwritten")

    def test_read_is_retried_and_nothing_is_lost(self):
        self._seed_two_pending()
        p_open, p_copy = self._locked()
        with p_open, p_copy:
            new_id = memory.make_promise("made while locked", "manual", source="t")
        # The lock is gone: the next call reads the file and keeps both sides.
        pending = memory.list_promises()
        self.assertEqual(sorted(p["message"] for p in pending),
                         ["made while locked", "pending one", "pending two"])
        ids = [p["id"] for p in pending]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate ids: {ids}")
        # The promise made while locked can still be cancelled by its id.
        made = [p for p in pending if p["message"] == "made while locked"][0]
        self.assertTrue(memory.cancel_promise(made["id"]))
        self.assertIsInstance(new_id, int)
        # ...and the next save persists all three.
        with open(self.promises_file, "r", encoding="utf-8") as f:
            self.assertEqual(
                sorted(p["message"] for p in json.load(f)),
                ["made while locked", "pending one", "pending two"])


if __name__ == "__main__":
    unittest.main()
