"""Regression: blue_green_manager._atomic_write_json must write the cross-process
instances.json via a UNIQUE per-call temp (mkstemp), NOT a fixed `path + ".tmp"`.

During a blue/green upgrade the prod (blue) and staging (green) JARVIS instances
BOTH heartbeat-write data/instances.json concurrently (plus the smoke driver), so
a shared temp name lets one writer truncate another's half-written temp -> a
0-byte / garbled instances.json that breaks role resolution and can derail the
handoff. Found by the full-codebase audit.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import blue_green_manager as bgm


class BlueGreenAtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.p = os.path.join(self.d, "instances.json")

    def test_roundtrip(self):
        self.assertTrue(bgm._atomic_write_json(self.p, {"a": 1, "b": 2}))
        with open(self.p, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1, "b": 2})
        self.assertFalse(os.path.exists(self.p + ".tmp"))

    def test_uses_unique_mkstemp_not_fixed_tmp(self):
        real = tempfile.mkstemp
        seen = []

        def spy(*a, **k):
            fd, name = real(*a, **k)
            seen.append(name)
            return fd, name

        with mock.patch("tempfile.mkstemp", side_effect=spy):
            self.assertTrue(bgm._atomic_write_json(self.p, {"x": 1}))
        self.assertEqual(len(seen), 1, "must use exactly one mkstemp temp")
        self.assertNotEqual(seen[0], self.p + ".tmp",
                            "must NOT use the fixed shared temp name")
        self.assertTrue(os.path.basename(seen[0]).startswith("instances.json."))

    def test_two_writers_get_distinct_temps(self):
        real = tempfile.mkstemp
        seen = []

        def spy(*a, **k):
            fd, name = real(*a, **k)
            seen.append(name)
            return fd, name

        with mock.patch("tempfile.mkstemp", side_effect=spy):
            bgm._atomic_write_json(self.p, {"w": 1})
            bgm._atomic_write_json(self.p, {"w": 2})
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1],
                            "concurrent writers must get distinct temps")


if __name__ == "__main__":
    unittest.main()
