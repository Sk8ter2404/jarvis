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


class WatchdogDiskEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def test_disk_free_gb_swallows_psutil_error(self):
        fake = mock.MagicMock()
        fake.disk_usage.side_effect = OSError("device gone")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._disk_free_gb())

    def test_check_disk_returns_when_reading_unavailable(self):
        # free_gb is None → early return, no alert (covers line 121).
        with mock.patch.object(self.mod, "_disk_free_gb", return_value=None), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_disk(time.time())
        enq.assert_not_called()


class WatchdogSnapshotEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def test_snapshot_malformed_json_returns_none_none(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("{ broken json ::::")
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertEqual(self.mod._read_credits_snapshot(), (None, None))

    def test_snapshot_missing_checked_at_gives_none_age(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"balance": 5.0}, f)  # no checked_at → age None
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            bal, age = self.mod._read_credits_snapshot()
        self.assertAlmostEqual(bal, 5.0)
        self.assertIsNone(age)


class WatchdogCreditsCheckEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def test_check_credits_returns_when_balance_none(self):
        # balance None → early return (covers line 136).
        with mock.patch.object(self.mod, "_read_credits_snapshot",
                               return_value=(None, None)), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_credits(time.time())
        enq.assert_not_called()


class WatchdogEnqueueSpeechTests(unittest.TestCase):
    """_enqueue_speech: proactive route + atomic-file fallback + write failure."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("disk_budget_watchdog")

    def test_enqueue_via_proactive_announce(self):
        fake_bc = mock.MagicMock()
        with mock.patch("importlib.import_module", return_value=fake_bc):
            self.mod._enqueue_speech("ann")
        fake_bc.proactive_announce.assert_called_once_with(
            "ann", source="disk_budget_watchdog")

    def test_enqueue_file_fallback_appends(self):
        bc_no_announce = mock.MagicMock(spec=[])
        with tempfile.TemporaryDirectory() as d:
            qpath = os.path.join(d, "pending_speech.json")
            with mock.patch("importlib.import_module", return_value=bc_no_announce), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath):
                self.mod._enqueue_speech("via-file")
            with open(qpath, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data[-1]["message"], "via-file")

    def test_enqueue_recovers_from_corrupt_existing_queue(self):
        with mock.patch("importlib.import_module", side_effect=ImportError):
            with tempfile.TemporaryDirectory() as d:
                qpath = os.path.join(d, "pending_speech.json")
                with open(qpath, "w", encoding="utf-8") as f:
                    f.write("not-json{")
                with mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath):
                    self.mod._enqueue_speech("recovered")
                with open(qpath, encoding="utf-8") as f:
                    data = json.load(f)
        self.assertEqual(data, [{"ts": mock.ANY, "message": "recovered"}])

    def test_enqueue_write_failure_prints_fallback(self):
        import contextlib
        import io as _io
        with mock.patch("importlib.import_module", side_effect=ImportError), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("full")), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False):
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.mod._enqueue_speech("doomed-alert")
        self.assertIn("doomed-alert", buf.getvalue())


class WatchdogRegisterTests(unittest.TestCase):
    """register(): the psutil-missing degradation notice (line 196)."""

    def test_register_warns_when_psutil_missing(self):
        import contextlib
        import io as _io
        import threading

        mod, _ = load_skill_isolated("disk_budget_watchdog", register=False)
        buf = _io.StringIO()
        # Neuter the monitor thread, force the psutil-missing branch, register.
        with mock.patch.object(threading.Thread, "start", lambda self: None), \
             mock.patch.object(mod, "_HAS_PSUTIL", False), \
             contextlib.redirect_stdout(buf):
            mod.register({})
        self.assertIn("psutil missing", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
