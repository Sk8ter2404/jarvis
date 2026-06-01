"""Logic tests for skills/stability_gate_status.py.

A pure read-only voice skill: it tails data/stability_gates.jsonl, parses the
freshest JSON line, and renders a one-sentence verdict summary
(PASS / FAIL / SKIP / unknown) with a human-friendly age phrase.

Everything is driven off a temp JSONL (via a patched _LOG_PATH) and a pinned
time.time so the age phrasing is deterministic. No real upgrade-pipeline log is
read or written.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class StabilityGateStatusTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("stability_gate_status")
        # Each test points _LOG_PATH at its own tempfile so nothing reads the
        # real data/stability_gates.jsonl.
        fd, self.log_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.log_path)
                        and os.remove(self.log_path))
        self._patch = mock.patch.object(self.mod, "_LOG_PATH", self.log_path)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _write_lines(self, *json_lines: str):
        with open(self.log_path, "w", encoding="utf-8") as f:
            for ln in json_lines:
                f.write(ln + "\n")

    # ── _read_last_record ────────────────────────────────────────────────
    def test_read_last_record_returns_freshest_line(self):
        self._write_lines('{"batch": 1, "verdict": "PASS"}',
                          '{"batch": 2, "verdict": "FAIL"}')
        rec = self.mod._read_last_record()
        self.assertEqual(rec["batch"], 2)
        self.assertEqual(rec["verdict"], "FAIL")

    def test_read_last_record_skips_trailing_blank_lines(self):
        # A trailing newline / blank line must not shadow the real last record.
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write('{"batch": 7, "verdict": "PASS"}\n\n   \n')
        rec = self.mod._read_last_record()
        self.assertEqual(rec["batch"], 7)

    def test_read_last_record_missing_file_is_none(self):
        os.remove(self.log_path)
        self.assertIsNone(self.mod._read_last_record())

    def test_read_last_record_bad_json_is_none(self):
        self._write_lines("{not valid json")
        self.assertIsNone(self.mod._read_last_record())

    # ── _friendly_age ────────────────────────────────────────────────────
    def test_friendly_age_just_now(self):
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        with mock.patch.object(self.mod.time, "time", return_value=now + 5):
            self.assertEqual(self.mod._friendly_age(iso), "just now")

    def test_friendly_age_minutes(self):
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        with mock.patch.object(self.mod.time, "time", return_value=now + 14 * 60):
            self.assertIn("14 minute", self.mod._friendly_age(iso))

    def test_friendly_age_hours(self):
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        with mock.patch.object(self.mod.time, "time", return_value=now + 3 * 3600):
            self.assertIn("3 hour", self.mod._friendly_age(iso))

    def test_friendly_age_unparseable_returns_input(self):
        self.assertEqual(self.mod._friendly_age("not-a-timestamp"),
                         "not-a-timestamp")

    # ── last_stability_gate (the action) ─────────────────────────────────
    def test_no_log_yet_is_graceful(self):
        os.remove(self.log_path)
        out = self.actions["stability_gate_status"]("")
        self.assertIn("haven't run a stability gate", out.lower())

    def test_pass_verdict_renders_batch_and_duration(self):
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        self._write_lines(
            '{"batch": 42, "verdict": "PASS", "duration_s": 18, "ts": "%s"}'
            % iso)
        with mock.patch.object(self.mod.time, "time", return_value=now + 30):
            out = self.actions["last_stability_gate"]("")
        self.assertIn("batch 42", out)
        self.assertIn("PASSED", out)
        self.assertIn("18-second", out)

    def test_fail_verdict_surfaces_symptom_and_revert(self):
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        self._write_lines(
            '{"batch": 9, "verdict": "FAIL", "smoke_error": '
            '"ImportError: boom", "ts": "%s"}' % iso)
        with mock.patch.object(self.mod.time, "time", return_value=now + 60):
            out = self.actions["last_gate_result"]("")
        self.assertIn("batch 9", out)
        self.assertIn("FAILED", out)
        self.assertIn("ImportError: boom", out)
        self.assertIn("auto-reverted", out)

    def test_fail_verdict_truncates_long_symptom(self):
        long_err = "x" * 500
        self._write_lines(
            '{"batch": 1, "verdict": "FAIL", "smoke_error": "%s"}' % long_err)
        out = self.actions["stability_gate_status"]("")
        # Symptom is capped at 200 chars; the full 500-char blob must not pass.
        self.assertNotIn("x" * 300, out)

    def test_fail_verdict_falls_back_to_stdout_tail(self):
        # No smoke_error → use smoke_stdout_tail.
        self._write_lines(
            '{"batch": 3, "verdict": "FAIL", '
            '"smoke_stdout_tail": "  traceback tail here  "}')
        out = self.actions["stability_gate_status"]("")
        self.assertIn("traceback tail here", out)

    def test_skip_verdict_includes_reason(self):
        self._write_lines(
            '{"batch": 5, "verdict": "SKIP", "reason": "no code changes"}')
        out = self.actions["stability_gate_status"]("")
        self.assertIn("SKIPPED", out)
        self.assertIn("no code changes", out)

    def test_unknown_verdict_is_passed_through(self):
        self._write_lines('{"batch": 11, "verdict": "WEIRD"}')
        out = self.actions["stability_gate_status"]("")
        self.assertIn("WEIRD", out)
        self.assertIn("batch 11", out)

    # ── alias wiring ─────────────────────────────────────────────────────
    def test_all_aliases_registered_to_same_handler(self):
        for name in ("last_stability_gate", "last_stability_gate_result",
                     "last_gate_result", "stability_gate_status",
                     "gate_status"):
            self.assertIn(name, self.actions)
        # They are the same callable (module-level alias binding).
        self.assertIs(self.actions["gate_status"],
                      self.actions["stability_gate_status"])


if __name__ == "__main__":
    unittest.main()
