"""Tests for core.legacy_memory — the flat bobert_memory.json store. Pins the
schema, the empty/corrupt fallbacks, the forward-migration of missing keys, and
the atomic save round-trip (the store is dumped into the prompt every turn, so a
truncated/corrupt write is a real outage). Uses configure() to point at a temp
file so the real store is never touched."""
import json
import os
import tempfile
import threading
import unittest

import core.legacy_memory as lm


class LegacyMemoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = os.path.join(self._tmp.name, "mem.json")
        self._orig_file, self._orig_lock = lm._MEMORY_FILE, lm._LOCK
        lm.configure(self._path, threading.RLock())

    def tearDown(self):
        lm._MEMORY_FILE, lm._LOCK = self._orig_file, self._orig_lock
        self._tmp.cleanup()

    def test_empty_memory_schema(self):
        m = lm._empty_memory()
        for k in ("first_meeting", "conversation_count", "facts", "projects",
                  "topics", "sessions", "last_used_phrase_by_intent"):
            self.assertIn(k, m)

    def test_load_missing_file_returns_empty(self):
        m = lm.load_memory()
        self.assertEqual(m["facts"], [])
        self.assertEqual(m["conversation_count"], 0)

    def test_save_then_load_roundtrip(self):
        m = lm._empty_memory()
        m["facts"].append("User likes lofi")
        m["conversation_count"] = 5
        lm.save_memory(m)
        loaded = lm.load_memory()
        self.assertEqual(loaded["facts"], ["User likes lofi"])
        self.assertEqual(loaded["conversation_count"], 5)

    def test_save_leaves_no_tempfiles(self):
        lm.save_memory(lm._empty_memory())
        leftovers = [f for f in os.listdir(self._tmp.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [], f"stray tempfiles: {leftovers}")

    def test_load_corrupt_returns_empty(self):
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        self.assertEqual(lm.load_memory()["facts"], [])

    def test_load_migrates_missing_keys(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"facts": ["x"]}, f)   # old schema: facts only
        m = lm.load_memory()
        self.assertEqual(m["facts"], ["x"])
        self.assertIn("projects", m)
        self.assertIn("sessions", m)
        self.assertIn("last_used_phrase_by_intent", m)


if __name__ == "__main__":
    unittest.main()
