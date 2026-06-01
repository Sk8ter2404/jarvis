"""Logic tests for skills/disk_budget_watchdog.py.

Overnight guardrail watching C: free space and the Claude credit balance.
We test:

  • _disk_free_gb — psutil read + graceful None when psutil is missing
  • _read_credits_snapshot — (balance, age) parsing, missing/malformed file
  • _check_disk — alert only below threshold, honouring the per-metric cooldown
  • _check_credits — alert below threshold, suppressed when the snapshot is
    stale or within cooldown
  • the check_budget action — disk + credit summary, stale-snapshot note,
    psutil-missing degradation

The monitor thread never runs (harness neuters threads). _enqueue_speech is
patched in every alert test so nothing reaches pending_speech.json, and the
credits state file is a tempfile.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _Disk:
    def __init__(self, free_gb):
        self.free = int(free_gb * 1024 ** 3)


class WatchdogDiskTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def test_disk_free_gb_reads_psutil(self):
        fake = mock.MagicMock()
        fake.disk_usage.return_value = _Disk(123.0)
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertAlmostEqual(self.mod._disk_free_gb(), 123.0, places=1)

    def test_disk_free_gb_none_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._disk_free_gb())

    def test_check_disk_alerts_below_threshold(self):
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=4.2), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._last_disk_alert_at[0] = 0.0
            self.mod._check_disk(time.time())
        enq.assert_called_once()
        self.assertIn("4.2 gigabytes free", enq.call_args.args[0])

    def test_check_disk_silent_above_threshold(self):
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=500.0), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_disk(time.time())
        enq.assert_not_called()

    def test_check_disk_respects_cooldown(self):
        now = time.time()
        self.mod._last_disk_alert_at[0] = now - 60  # alerted a minute ago
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=2.0), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_disk(now)
        enq.assert_not_called()


class WatchdogCreditsSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def _write(self, payload):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        return p

    def test_snapshot_balance_and_age(self):
        p = self._write({"balance": 7.5, "checked_at": time.time() - 100})
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            bal, age = self.mod._read_credits_snapshot()
        self.assertAlmostEqual(bal, 7.5)
        self.assertGreaterEqual(age, 99)

    def test_snapshot_missing_file(self):
        with mock.patch.object(self.mod, "_CREDITS_STATE",
                               os.path.join(tempfile.gettempdir(),
                                            "no_such_watchdog.json")):
            self.assertEqual(self.mod._read_credits_snapshot(), (None, None))

    def test_snapshot_malformed_balance(self):
        p = self._write({"balance": "lots", "checked_at": time.time()})
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            bal, age = self.mod._read_credits_snapshot()
        self.assertIsNone(bal)
        self.assertIsNotNone(age)


class WatchdogCreditsCheckTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def test_alerts_low_fresh_balance(self):
        with mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(3.0, 100.0)), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._last_credits_alert_at[0] = 0.0
            self.mod._check_credits(time.time())
        enq.assert_called_once()
        self.assertIn("3.00 dollars", enq.call_args.args[0])

    def test_silent_when_balance_healthy(self):
        with mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(50.0, 100.0)), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_credits(time.time())
        enq.assert_not_called()

    def test_silent_when_snapshot_stale(self):
        # Balance is low but the snapshot is older than the stale window.
        with mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(1.0, 25 * 3600)), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_credits(time.time())
        enq.assert_not_called()

    def test_silent_within_cooldown(self):
        now = time.time()
        self.mod._last_credits_alert_at[0] = now - 60
        with mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(1.0, 100.0)), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_credits(now)
        enq.assert_not_called()


class WatchdogActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def test_check_budget_reports_disk_and_credits(self):
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=42.0), \
             mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(15.0, 100.0)):
            out = self.actions["check_budget"]("")
        self.assertIn("42.0 gigabytes free", out)
        self.assertIn("$15.00", out)
        self.assertNotIn("stale", out)

    def test_check_budget_marks_stale_snapshot(self):
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=42.0), \
             mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(15.0, 25 * 3600)):
            out = self.actions["check_budget"]("")
        self.assertIn("stale", out.lower())

    def test_check_budget_no_credits_recorded(self):
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=42.0), \
             mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(None, None)):
            out = self.actions["check_budget"]("")
        self.assertIn("no Claude credit balance recorded", out)

    def test_check_budget_disk_unavailable(self):
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=None), \
             mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(None, None)):
            out = self.actions["check_budget"]("")
        self.assertIn("disk reading unavailable", out)


if __name__ == "__main__":
    unittest.main()
