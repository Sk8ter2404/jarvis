"""Logic tests for skills/timer.py.

Exemplar for per-skill logic tests: drive registered actions through the
isolation harness (no monolith), control threads so no real timers spawn, and
mock the cross-module speech enqueue. Covers pure functions, error paths,
happy paths + state, and the blue/green restore path.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated, no_background_threads


class TimerSkillTests(unittest.TestCase):
    def setUp(self):
        # Fresh module + actions per test; timer keeps module-global state, and
        # re-exec gives a clean _timers/_next_id, but reset defensively.
        self.mod, self.actions = load_skill_isolated("timer")
        self.mod._timers.clear()
        self.mod._next_id[0] = 1

    # ── _parse_duration (pure) ───────────────────────────────────────────
    def test_parse_duration_units(self):
        p = self.mod._parse_duration
        self.assertEqual(p("30 seconds"), 30)
        self.assertEqual(p("5 minutes"), 300)
        self.assertEqual(p("2 hours"), 7200)
        self.assertEqual(p("1 day"), 86400)
        self.assertEqual(p("90 secs"), 90)
        self.assertEqual(p("1 hour 30 minutes"), 5400)  # compound

    def test_parse_duration_invalid(self):
        self.assertIsNone(self.mod._parse_duration("soon"))
        self.assertIsNone(self.mod._parse_duration(""))

    # ── set_timer error paths (no thread started) ────────────────────────
    def test_set_timer_requires_pipe(self):
        self.assertIn("format", self.actions["set_timer"]("5 minutes oops").lower())

    def test_set_timer_bad_duration(self):
        self.assertIn("could not parse",
                      self.actions["set_timer"]("whenever | check oven").lower())

    def test_set_timer_empty_message(self):
        self.assertIn("message", self.actions["set_timer"]("5 minutes | ").lower())

    # ── set_timer happy path (threads neutered) ──────────────────────────
    def test_set_timer_happy(self):
        with no_background_threads():
            out = self.actions["set_timer"]("5 minutes | check the oven")
        self.assertIn("#1", out)
        self.assertIn("5m", out)
        self.assertIn("check the oven", out)
        self.assertEqual(len(self.mod._timers), 1)

    def test_list_and_cancel(self):
        with no_background_threads():
            self.actions["set_timer"]("10 minutes | tea")
            self.actions["set_timer"]("20 minutes | walk")
        listed = self.actions["list_timers"]()
        self.assertIn("2 active timer", listed)
        self.assertIn("tea", listed)

        self.assertIn("cancelled timer #1", self.actions["cancel_timer"]("1"))
        self.assertEqual(len(self.mod._timers), 1)
        self.assertIn("cancelled", self.actions["cancel_timer"]("all"))
        self.assertEqual(len(self.mod._timers), 0)

    def test_cancel_unknown(self):
        self.assertIn("no timer", self.actions["cancel_timer"]("999"))

    def test_list_empty(self):
        self.assertEqual(self.actions["list_timers"](), "no active timers")

    # ── restore_timers (blue/green handoff) ──────────────────────────────
    def test_restore_past_timer_fires_immediately(self):
        with mock.patch.object(self.mod, "_enqueue_speech") as enq:
            n = self.mod.restore_timers(
                [{"id": 5, "message": "stretch", "fire_at": 1.0}])
        self.assertEqual(n, 1)
        enq.assert_called_once()
        self.assertIn("stretch", enq.call_args[0][0])

    def test_restore_future_timer_rearms(self):
        with no_background_threads():
            n = self.mod.restore_timers(
                [{"id": 7, "message": "later", "fire_at": time.time() + 9999}])
        self.assertEqual(n, 1)
        self.assertIn(7, self.mod._timers)

    def test_restore_rejects_garbage(self):
        self.assertEqual(self.mod.restore_timers("not a list"), 0)
        self.assertEqual(self.mod.restore_timers([{"bad": "entry"}]), 0)


if __name__ == "__main__":
    unittest.main()
