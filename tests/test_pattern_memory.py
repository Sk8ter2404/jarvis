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


class ConversationLogWipeTests(unittest.TestCase):
    """B003 (2026-10-01): "forget the last hour" and "reset your memory" left
    memory/session_summaries.json alone ("what did we do this afternoon"
    recited the forgotten hour, and the 10-minute checkpoint re-wrote it),
    and a reset also left the verbatim voice-command log."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.sessions = os.path.join(self._tmp.name, "session_summaries.json")
        self.log = os.path.join(self._tmp.name, "voice_commands.jsonl")
        for name, path in (("_SESSION_FILE", self.sessions),
                           ("_LOG_FILE", self.log),
                           ("_MEM_DIR", self._tmp.name)):
            p = mock.patch.object(pattern_memory, name, path)
            p.start()
            self.addCleanup(p.stop)

    def _write_sessions(self, entries):
        with open(self.sessions, "w", encoding="utf-8") as f:
            json.dump(entries, f)

    def _read_sessions(self):
        with open(self.sessions, encoding="utf-8") as f:
            return json.load(f)

    def test_forget_drops_summaries_that_reach_into_the_window(self):
        now = time.time()
        today = time.strftime("%Y-%m-%d")
        self._write_sessions([
            {"ts": now - 7200, "date": today,
             "summary": "morning: the garden"},
            {"ts": now - 600, "date": today, "summary": "booked the dentist"},
            {"iso_end": time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(now - 60)),
             "date": today, "summary": "no float ts, recent iso_end"},
            {"summary": "legacy entry with no end time"},
        ])
        dropped = pattern_memory.forget_session_summaries_since(now - 3600)
        self.assertEqual(dropped, 2)
        left = [e["summary"] for e in self._read_sessions()]
        self.assertEqual(left, ["morning: the garden",
                                "legacy entry with no end time"])
        self.assertFalse(os.path.exists(self.sessions + ".tmp"))
        # Recall no longer finds the forgotten hour.
        got = [e.get("summary") for e in pattern_memory.get_session_summaries(
            "today", include_legacy=False)]
        self.assertIn("morning: the garden", got)
        self.assertNotIn("booked the dentist", got)

    def test_forget_with_nothing_in_the_window_rewrites_nothing(self):
        now = time.time()
        self._write_sessions([{"ts": now - 7200, "summary": "old"}])
        before = os.path.getmtime(self.sessions)
        self.assertEqual(
            pattern_memory.forget_session_summaries_since(now - 3600), 0)
        self.assertEqual(os.path.getmtime(self.sessions), before)
        self.assertEqual(
            pattern_memory.forget_session_summaries_since(now - 3600), 0)

    def test_forget_failures_propagate_for_disclosure(self):
        with open(self.sessions, "w", encoding="utf-8") as f:
            f.write("{ not json")
        with self.assertRaises(ValueError):
            pattern_memory.forget_session_summaries_since(0.0)

    def test_reset_backs_up_then_clears_both_logs(self):
        self._write_sessions([{"ts": 1.0, "summary": "a"},
                              {"ts": 2.0, "summary": "b"}])
        with open(self.log, "w", encoding="utf-8") as f:
            f.write('{"ts": 1, "text": "x"}\n{"text": "legacy, no ts"}\n')
        backups = os.path.join(self._tmp.name, "backups")
        counts = pattern_memory.reset_conversation_logs(backups)
        self.assertEqual(counts, {"session_summaries": 2,
                                  "voice_commands": 2})
        self.assertEqual(self._read_sessions(), [])
        with open(self.log, encoding="utf-8") as f:
            self.assertEqual(f.read(), "")       # ts-less lines too
        saved = sorted(os.listdir(backups))
        self.assertEqual(len(saved), 2)
        self.assertTrue(all(n.startswith("pre_reset_") for n in saved))

    def test_reset_clears_nothing_when_the_backup_fails(self):
        self._write_sessions([{"ts": 1.0, "summary": "a"}])
        with mock.patch.object(pattern_memory.shutil, "copy2",
                               side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                pattern_memory.reset_conversation_logs(
                    os.path.join(self._tmp.name, "backups"))
        self.assertEqual(len(self._read_sessions()), 1)


if __name__ == "__main__":
    unittest.main()
