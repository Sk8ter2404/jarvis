"""Tests for the root ``memory.py`` (pattern memory) voice-command purge.

2026-07-21 audit #51: 'forget the last hour' pruned bobert_memory.json and
(after the fix) the tiered LTM store, but memory/voice_commands.jsonl still
held the user's verbatim last-hour speech — the identical failure one store
over. ``forget_voice_commands_since`` closes that gap; these tests pin its
contract: time-window drop, atomic tmp+os.replace rewrite, legacy (ts-less /
unparseable) lines kept, exceptions propagate so the caller can DISCLOSE a
failed purge.

Every test redirects ``_LOG_FILE`` into a per-test tempdir so the real
(staging or live) log is never touched. stdlib unittest + mock only.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

import memory as pattern_memory


class ForgetVoiceCommandsSinceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = os.path.join(self._tmp.name, "voice_commands.jsonl")
        patcher = mock.patch.object(pattern_memory, "_LOG_FILE", self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_lines(self, lines):
        with open(self.log, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write((json.dumps(ln) if isinstance(ln, dict) else ln)
                        + "\n")

    def test_purges_recent_keeps_old_and_legacy(self):
        now = time.time()
        self._write_lines([
            {"ts": now - 7200, "text": "old command"},          # kept
            {"ts": now - 60,   "text": "the private one"},      # dropped
            "not json at all",                                  # kept (legacy)
            {"text": "no ts entry"},                            # kept (legacy)
        ])
        dropped = pattern_memory.forget_voice_commands_since(now - 3600)
        self.assertEqual(dropped, 1)
        with open(self.log, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("old command", content)
        self.assertIn("not json at all", content)
        self.assertIn("no ts entry", content)
        self.assertNotIn("the private one", content)
        # Atomic rewrite: no tempfile residue.
        self.assertFalse(os.path.exists(self.log + ".tmp"))

    def test_all_recent_drops_everything(self):
        now = time.time()
        self._write_lines([
            {"ts": now - 10, "text": "one"},
            {"ts": now - 20, "text": "two"},
        ])
        self.assertEqual(
            pattern_memory.forget_voice_commands_since(now - 3600), 2)
        with open(self.log, encoding="utf-8") as f:
            self.assertEqual(f.read(), "")

    def test_no_recent_entries_leaves_file_untouched(self):
        now = time.time()
        self._write_lines([{"ts": now - 7200, "text": "old"}])
        self.assertEqual(
            pattern_memory.forget_voice_commands_since(now - 3600), 0)
        with open(self.log, encoding="utf-8") as f:
            self.assertIn("old", f.read())
        self.assertFalse(os.path.exists(self.log + ".tmp"))

    def test_missing_file_returns_zero(self):
        self.assertEqual(
            pattern_memory.forget_voice_commands_since(time.time()), 0)

    def test_read_failure_propagates_for_disclosure(self):
        # The caller (_act_forget_last_hour) must be able to disclose a failed
        # purge — a swallowed exception here would reintroduce the silent-
        # survival gap the 2026-07-21 audit flagged.
        self._write_lines([{"ts": time.time(), "text": "x"}])
        with mock.patch("builtins.open",
                        side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                pattern_memory.forget_voice_commands_since(0.0)


if __name__ == "__main__":
    unittest.main()
