"""Tests for core.atomic_io — the shared atomic JSON writer that every
state-file writer in JARVIS relies on. If this regresses, concurrent readers
(HUD, overlays, skills) can see half-written JSON, so it's load-bearing."""
import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
