"""Tests for the overnight upgrade engine's stability circuit breaker — the
counter/trip/cooldown logic that stops the engine from churning (and burning
API spend) on the same failing upgrade night after night. Pure state-dict
logic, so no subprocess / network / filesystem needed."""
import unittest

import overnight_upgrade as ovn


class RecordOutcomeTests(unittest.TestCase):
    def test_none_leaves_counter_unchanged(self):
        st = {"consecutive_failures": 2}
        ovn._record_cycle_outcome(st, None, now=1000.0)
        self.assertEqual(st["consecutive_failures"], 2)

    def test_productive_resets(self):
        st = {"consecutive_failures": 2}
        ovn._record_cycle_outcome(st, True, now=1000.0)
        self.assertEqual(st["consecutive_failures"], 0)

    def test_failure_increments_without_tripping_early(self):
        st = {}
        ovn._record_cycle_outcome(st, False, now=1000.0)
        self.assertEqual(st["consecutive_failures"], 1)
        self.assertNotIn("breaker_tripped_at", st)

    def test_trips_at_threshold(self):
        st = {"consecutive_failures": ovn.MAX_CONSECUTIVE_FAILURES - 1}
        ovn._record_cycle_outcome(st, False, now=1234.0)
        self.assertEqual(st["consecutive_failures"], ovn.MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(st["breaker_tripped_at"], 1234.0)

    def test_productive_clears_a_trip(self):
        st = {"consecutive_failures": 9, "breaker_tripped_at": 999.0}
        ovn._record_cycle_outcome(st, True, now=1000.0)
        self.assertEqual(st["consecutive_failures"], 0)
        self.assertNotIn("breaker_tripped_at", st)


class SkipReasonTests(unittest.TestCase):
    def test_no_trip_means_no_skip(self):
        self.assertIsNone(ovn._breaker_skip_reason({}, now=1000.0))

    def test_within_cooldown_skips(self):
        now = 1_000_000.0
        st = {"consecutive_failures": ovn.MAX_CONSECUTIVE_FAILURES,
              "breaker_tripped_at": now - 3600.0}   # tripped 1h ago
        reason = ovn._breaker_skip_reason(st, now)
        self.assertIsNotNone(reason)
        self.assertIn("circuit breaker", reason.lower())
        # still tripped — not auto-reset yet
        self.assertIn("breaker_tripped_at", st)

    def test_after_cooldown_autoresets_and_resumes(self):
        now = 1_000_000.0
        cooldown_s = ovn.BREAKER_COOLDOWN_HOURS * 3600.0
        st = {"consecutive_failures": ovn.MAX_CONSECUTIVE_FAILURES,
              "breaker_tripped_at": now - cooldown_s - 10.0}
        reason = ovn._breaker_skip_reason(st, now)
        self.assertIsNone(reason)                       # resumes
        self.assertEqual(st["consecutive_failures"], 0)  # auto-reset
        self.assertNotIn("breaker_tripped_at", st)


if __name__ == "__main__":
    unittest.main()
