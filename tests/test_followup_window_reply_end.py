"""core/followup_window.py 3.0.26: the window runs from the END of the reply.

Live 11-16..11-17 (wake-word mode, FOLLOWUP_WINDOW_S 20): the window opened
at the wake gate, before the brain and the reply, so with the median voice
turn ending 9.8 s after he stopped talking most follow-ups had 12 s or less,
and 5 of 55 turns had closed the window before JARVIS stopped speaking.
rearm_after_reply() restarts the window when the reply has been played - for
a turn that opened it or kept its extension only.

Pure: no monolith.

    python tools/run_tests.py test_followup_window_reply_end
"""
from __future__ import annotations

import unittest

from core.followup_window import FollowupWindow

WINDOW = 20.0     # the owner's FOLLOWUP_WINDOW_S


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class RearmTests(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock(100.0)
        self.win = FollowupWindow(WINDOW, clock=self.clock)

    def test_a_wake_turn_re_arms_from_the_end_of_a_long_reply(self):
        self.win.note_addressed()                 # gate at t=100
        self.clock.t = 128.0                      # a 28 s reply ends
        self.assertFalse(self.win.remaining_s())  # closed while he spoke
        self.assertEqual(self.win.rearm_after_reply(), 0.0)
        self.assertAlmostEqual(self.win.remaining_s(), WINDOW)
        self.clock.t = 133.0                      # his answer 5 s later
        self.assertTrue(self.win.admit())

    def test_without_the_re_arm_that_answer_is_refused(self):
        self.win.note_addressed()
        self.clock.t = 133.0
        self.assertFalse(self.win.admit())

    def test_the_re_arm_never_shortens_the_window(self):
        self.win.note_addressed()                 # until 120
        self.win._until = 150.0                   # already further out
        self.clock.t = 101.0
        self.win.rearm_after_reply()
        self.assertEqual(self.win._until, 150.0)

    def test_the_mark_is_consumed_once(self):
        self.win.note_addressed()
        self.clock.t = 110.0
        self.assertIsNotNone(self.win.rearm_after_reply())
        self.clock.t = 125.0
        self.assertIsNone(self.win.rearm_after_reply())
        self.assertAlmostEqual(self.win.remaining_s(), 5.0)

    def test_a_kept_extension_re_arms(self):
        self.win.note_addressed()
        self.win.rearm_after_reply()
        self.clock.t = 105.0
        self.assertTrue(self.win.admit())
        self.win.forget_admit()                   # its voice was confirmed
        self.clock.t = 130.0                      # its reply ends
        self.assertEqual(self.win.rearm_after_reply(), 0.0)
        self.assertAlmostEqual(self.win.remaining_s(), WINDOW)

    def test_a_taken_back_extension_never_re_arms(self):
        self.win.note_addressed()
        self.win.rearm_after_reply()              # until 120
        self.clock.t = 105.0
        self.assertTrue(self.win.admit())         # pushed to 125
        self.assertTrue(self.win.take_back_admit())   # unsure voice: 120 again
        self.assertFalse(self.win.reply_due)
        self.clock.t = 112.0
        self.assertIsNone(self.win.rearm_after_reply())
        self.assertAlmostEqual(self.win.remaining_s(), 8.0)

    def test_a_dropped_turn_leaves_the_take_back_intact(self):
        self.win.note_addressed()
        self.win.rearm_after_reply()              # until 120
        self.clock.t = 105.0
        self.assertTrue(self.win.admit())         # noise admitted: 125
        self.win.drop_reply_due()                 # the filter dropped it
        self.assertTrue(self.win.take_back_admit())   # the next gate
        self.assertEqual(self.win._until, 120.0)

    def test_a_refused_line_clears_the_mark(self):
        self.clock.t = 500.0                      # the window is closed
        self.assertFalse(self.win.admit())
        self.assertIsNone(self.win.rearm_after_reply())
        self.assertFalse(self.win.remaining_s())

    def test_a_disabled_window_never_re_arms(self):
        win = FollowupWindow(0, clock=self.clock)
        win.note_addressed()
        self.assertFalse(win.reply_due)
        self.assertIsNone(win.rearm_after_reply())
        self.assertFalse(win.admit())


if __name__ == "__main__":
    unittest.main()
