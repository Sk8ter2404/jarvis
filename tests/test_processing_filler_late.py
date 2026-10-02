"""A late stage-1 filler is skipped, not played over the answer (NEW #8, 2026-10-02).

Live 22:04:11 (2026-10-01): with PROCESSING_FILLER_DELAY 0.5 s the stage-1
"I heard you" line slipped to +3.6 s - the scheduler kept retrying it for
first_retry_s (3 s) while a capture held it off - and because a started clip
always finishes, the answer then waited about 0.9 s behind it.

ProcessingFiller(first_late_s=...) caps how late stage 1 may START: past
t0 + first + first_late_s it is skipped (logged through log_fn), and
ack_state_here stops calling it 'pending' at the same moment. Default None
keeps the old behaviour for other callers; the monolith passes 1.0 s
(tests/monolith/test_monolith_boot_prime_retry.py checks the wiring).

Light tier: the FakeClock harness of tests/test_processing_filler.py.

    python -m unittest tests.test_processing_filler_late
"""
from __future__ import annotations

import threading
import unittest

from core import processing_filler as pf
from tests.test_processing_filler import FakeClock, RecFactory


def _make(first=0.5, still=12.0, late=1.0, logs=None):
    clock = FakeClock()
    plays: list = []
    holder: dict = {}

    def play_fn(turn, stage):
        v = holder["f"].claim(turn, stage)
        if v in ("not-yet", "busy"):
            return "retry"
        if v != "ok":
            return "skipped"
        plays.append((stage, clock.now))
        holder["f"].play_done()
        return "played"

    kwargs = dict(play_fn=play_fn, suppressed_fn=lambda: None,
                  delays_fn=lambda: (first, still), clock=clock,
                  wait_fn=clock.wait, thread_factory=RecFactory())
    try:
        f = pf.ProcessingFiller(first_late_s=late,
                                log_fn=(logs.append if logs is not None
                                        else None), **kwargs)
    except TypeError:
        f = pf.ProcessingFiller(**kwargs)     # pre-fix: no lateness cap
    holder["f"] = f
    return f, clock, plays


def _background_capture(f, clock, start, end):
    def on_other_thread(fn):
        th = threading.Thread(target=fn)
        th.start()
        th.join()
    clock.at(start, lambda: on_other_thread(f.begin_capture))
    clock.at(end, lambda: on_other_thread(f.end_capture))


class LateStageOneTests(unittest.TestCase):

    def test_a_capture_holding_stage_one_for_3s_skips_it(self):
        # The 22:04 shape: due at 0.5 s, held off until ~3.1 s; the old
        # 3 s retry window then played it at 3.5 s, over the answer.
        logs: list = []
        f, clock, plays = _make(first=0.5, late=1.0, logs=logs)
        t = f.arm()
        t.owner = -1                       # the capture is not this turn's
        _background_capture(f, clock, 0.2, 3.1)
        f._run(t)
        self.assertNotIn(1, [s for s, _ in plays],
                         f"stage 1 played late: {plays}")
        self.assertTrue(any("late" in m for m in logs), logs)

    def test_a_short_hold_still_plays_within_the_cap(self):
        f, clock, plays = _make(first=0.5, late=1.0)
        t = f.arm()
        t.owner = -1
        _background_capture(f, clock, 0.2, 1.2)
        f._run(t)
        firsts = [ts for s, ts in plays if s == 1]
        self.assertEqual(len(firsts), 1)
        self.assertLessEqual(firsts[0], 0.5 + 1.0 + 1e-9)

    def test_on_time_is_unchanged(self):
        f, clock, plays = _make(first=0.5, late=1.0)
        t = f.arm()
        f._run(t)
        self.assertEqual(plays[0], (1, 0.5))

    def test_a_late_wake_up_is_skipped_too(self):
        # The thread itself woke late (CPU busy decoding): no retry involved.
        f, clock, plays = _make(first=0.5, late=1.0)
        t = f.arm()
        t.t0 -= 2.0          # the filler thread got the CPU 2 s after arm()
        f._run(t)
        self.assertNotIn(1, [s for s, _ in plays])

    def test_ack_state_pending_ends_with_the_cap(self):
        f, clock, _p = _make(first=0.5, late=1.0)
        f.arm()
        clock.now = 1.4
        self.assertEqual(f.ack_state_here(), "pending")
        clock.now = 1.6
        self.assertEqual(f.ack_state_here(), "",
                         "a stage 1 that will never play is still 'pending'")

    def test_stage_two_is_unaffected(self):
        f, clock, plays = _make(first=0.5, still=12.0, late=1.0)
        t = f.arm()
        t.owner = -1
        _background_capture(f, clock, 0.2, 3.6)
        f._run(t)
        self.assertIn(2, [s for s, _ in plays])

    def test_default_keeps_the_old_window(self):
        f = pf.ProcessingFiller(play_fn=lambda t, s: "played",
                                suppressed_fn=lambda: None,
                                delays_fn=lambda: (2.5, 12.0))
        self.assertIsNone(getattr(f, "_first_late_s", None))


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
