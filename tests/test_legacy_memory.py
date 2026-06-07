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
from unittest import mock

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

    def test_load_coerces_string_list_fields_to_empty_list(self):
        # A corrupted store where a list field is a STRING must NOT survive as a
        # char-iterable: build_system_prompt does `for f in mem["facts"]`, so a
        # stray string would be iterated character-by-character into the cloud
        # prompt (token bloat / garbage). load_memory resets it to [].
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"facts": "User likes lofi",      # should have been a list
                       "projects": "Building a robot",
                       "topics": "weather"}, f)
        m = lm.load_memory()
        for k in ("facts", "projects", "topics"):
            self.assertIsInstance(m[k], list, f"{k} not coerced to list")
            self.assertEqual(m[k], [], f"{k} should reset to [], got {m[k]!r}")
        # Specifically: NOT the char-by-char explosion of the original string.
        self.assertNotIn("U", m["facts"])

    def test_load_coerces_non_dict_phrase_map_to_empty_dict(self):
        # last_used_phrase_by_intent feeds render_phrasebook_block(); a stray
        # non-dict (e.g. a string) would break it, so coerce to {}.
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"last_used_phrase_by_intent": "greeting"}, f)
        m = lm.load_memory()
        self.assertEqual(m["last_used_phrase_by_intent"], {})

    def test_load_preserves_valid_list_fields(self):
        # Coercion must not disturb already-valid data.
        good = {"facts": ["a", "b"], "projects": ["p"],
                "topics": [{"date": "2026-01-01", "topic": "t"}]}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(good, f)
        m = lm.load_memory()
        self.assertEqual(m["facts"], ["a", "b"])
        self.assertEqual(m["projects"], ["p"])
        self.assertEqual(m["topics"], [{"date": "2026-01-01", "topic": "t"}])

    def test_save_failure_cleans_tempfile_and_reraises(self):
        # If serialisation fails mid-write, save_memory must (a) re-raise so the
        # caller sees the failure and (b) leave NO stray .mem_*.tmp behind — the
        # atomic-write contract: the live store is never touched, and we don't
        # litter the data dir with half-written temp files.
        boom = RuntimeError("disk full")
        with mock.patch.object(lm.json, "dump", side_effect=boom):
            with self.assertRaises(RuntimeError):
                lm.save_memory(lm._empty_memory())
        leftovers = [f for f in os.listdir(self._tmp.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [], f"stray tempfiles: {leftovers}")
        # The original target was never created (no successful os.replace).
        self.assertFalse(os.path.exists(self._path))

    def test_save_failure_tempfile_cleanup_error_is_suppressed(self):
        # Belt-and-braces: even if the temp-file cleanup ALSO fails (os.remove
        # raises), save_memory still re-raises the ORIGINAL write error — the
        # cleanup failure is swallowed and never masks the real cause.
        boom = RuntimeError("disk full")
        with mock.patch.object(lm.json, "dump", side_effect=boom), \
             mock.patch.object(lm.os, "remove",
                               side_effect=PermissionError("locked")):
            with self.assertRaises(RuntimeError) as cm:
                lm.save_memory(lm._empty_memory())
        self.assertIs(cm.exception, boom)


if __name__ == "__main__":
    unittest.main()
