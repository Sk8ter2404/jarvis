"""core/input_backoff.py — the pacing half of the R10 mic-outage fix.

Live 2026-09-29 18:38:33: with no input device, record_speech failed ~200
times a second (5,859 failures, 11,718 tracebacks, a 20.7 MB log) because
nothing slowed its retries down. InputOpenBackoff is that missing pacing:
0.5 -> 1 -> 2 -> 5 s between attempts, reset by a successful open, plus the
bookkeeping for one-line-per-episode logging. Pure: every method takes the
caller's clock, so these tests drive it by hand.

    python -m unittest tests.test_input_backoff
"""
from __future__ import annotations

import unittest

from core.input_backoff import (DEFAULT_STEPS_S, DEFAULT_SUMMARY_S,
                                InputOpenBackoff)

T0 = 1000.0


class ScheduleTests(unittest.TestCase):
    def test_healthy_is_always_due(self):
        b = InputOpenBackoff()
        self.assertFalse(b.active)
        self.assertTrue(b.due(T0))
        self.assertEqual(b.remaining(T0), 0.0)

    def test_steps_are_half_one_two_then_capped_at_five(self):
        b = InputOpenBackoff()
        now, gaps = T0, []
        for _ in range(7):
            st = b.note_failure(now, "Error querying device -1", no_device=True)
            gaps.append(st["delay"])
            self.assertFalse(b.due(now))
            self.assertAlmostEqual(b.remaining(now), st["delay"])
            now += st["delay"]
            self.assertTrue(b.due(now))
        self.assertEqual(gaps, [0.5, 1.0, 2.0, 5.0, 5.0, 5.0, 5.0])
        self.assertEqual(DEFAULT_STEPS_S, (0.5, 1.0, 2.0, 5.0))

    def test_success_resets_to_the_first_step(self):
        b = InputOpenBackoff()
        now = T0
        for _ in range(5):
            now += b.note_failure(now, "x")["delay"]
        info = b.note_success(now)
        self.assertEqual(info["attempts"], 5)
        self.assertAlmostEqual(info["elapsed"], 0.5 + 1 + 2 + 5 + 5)
        self.assertFalse(b.active)
        self.assertTrue(b.due(now))
        self.assertIsNone(b.note_success(now), "no episode, nothing to end")
        self.assertEqual(b.note_failure(now, "x")["delay"], 0.5)

    def test_expedite_makes_the_next_attempt_due_now(self):
        b = InputOpenBackoff()
        for _ in range(4):
            b.note_failure(T0, "x")
        self.assertFalse(b.due(T0 + 1.0))
        b.expedite()
        self.assertTrue(b.due(T0))

    def test_pull_in_only_moves_earlier_and_never_below_the_first_step(self):
        b = InputOpenBackoff()
        for _ in range(4):
            b.note_failure(T0, "x")             # next at T0 + 5
        b.pull_in(T0, 9.0)                      # later: ignored
        self.assertAlmostEqual(b.remaining(T0), 5.0)
        b.pull_in(T0, 1.5)
        self.assertAlmostEqual(b.remaining(T0), 1.5)
        b.pull_in(T0, 0.0)                      # a hot loop is impossible
        self.assertAlmostEqual(b.remaining(T0), 0.5)
        healthy = InputOpenBackoff()
        healthy.pull_in(T0, 1.0)
        self.assertTrue(healthy.due(T0))

    def test_bad_constructor_values_fall_back(self):
        b = InputOpenBackoff(steps=("x", -1, 0), summary_s="soon")
        self.assertEqual(b.steps, DEFAULT_STEPS_S)
        self.assertEqual(b.summary_s, DEFAULT_SUMMARY_S)
        self.assertEqual(InputOpenBackoff(steps=(2, 3)).delay_after(9), 3.0)


class LogCadenceTests(unittest.TestCase):
    def test_one_first_line_then_a_summary_a_minute(self):
        b = InputOpenBackoff()
        logs, now = [], T0
        while now < T0 + 130:
            st = b.note_failure(now, "Error querying device -1", True)
            logs.append((round(now - T0, 1), st["log"]))
            now += st["delay"]
        marked = [(t, k) for t, k in logs if k]
        self.assertEqual(marked[0], (0.0, "first"))
        self.assertEqual([k for _, k in marked[1:]], ["summary", "summary"])
        self.assertGreaterEqual(marked[1][0], 60.0)
        self.assertLess(marked[1][0], 65.0)
        self.assertGreaterEqual(marked[2][0] - marked[1][0], 60.0)
        # ...and nothing in between: ~30 attempts, three lines.
        self.assertGreater(len(logs), 25)

    def test_a_change_of_failure_class_is_logged_again(self):
        b = InputOpenBackoff()
        self.assertEqual(b.note_failure(T0, "-9999", False)["log"], "first")
        self.assertIsNone(b.note_failure(T0 + 1, "-9999", False)["log"])
        self.assertEqual(b.note_failure(T0 + 2, "-1", True)["log"], "first")
        self.assertIsNone(b.note_failure(T0 + 3, "-1", True)["log"])

    def test_summary_reports_attempts_and_elapsed(self):
        b = InputOpenBackoff(summary_s=10)
        b.note_failure(T0, "x", True)
        self.assertIsNone(b.note_failure(T0 + 5, "x", True)["log"])
        st = b.note_failure(T0 + 10, "x", True)
        self.assertEqual(st["log"], "summary")
        self.assertEqual(st["attempts"], 3)
        self.assertAlmostEqual(st["elapsed"], 10.0)
        self.assertTrue(st["no_device"])


if __name__ == "__main__":
    unittest.main()
