"""core/followup_window.py take-back + core/media_gate.keeps_followup_extension
(3.0.6, scope B5).

Live 2026-10-28 17:28: "play vision quest on the left monitor" opened the
wake-word follow-up window (FOLLOWUP_WINDOW_S 20), a "Yeah." heard over the
YouTube video he had just started (voice 0.64, unsure) was admitted and
extended it, and the video's own 164-character narration 18 s later (0.68)
was admitted too and ran session_memory_recall.

Pure: no monolith, an injected clock.

    python -m unittest tests.test_followup_window_media
"""
from __future__ import annotations

import unittest

from core import learn_gate as lg
from core import media_gate as mg
from core.followup_window import FollowupWindow


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class TakeBackTests(unittest.TestCase):
    def test_the_live_1728_sequence_closes_on_the_original_deadline(self):
        c = _Clock(0.0)
        w = FollowupWindow(20, clock=c)
        w.note_addressed()                     # 17:28:11 "Jarvis, play vision quest ..."
        c.t = 16.0
        w.forget_admit()                       # the gate's new decision
        self.assertTrue(w.admit())             # 17:28:27 "Yeah." (over the video)
        self.assertTrue(w.take_back_admit())   # unsure over media: not extended
        self.assertAlmostEqual(w.remaining_s(), 4.0)
        c.t = 34.0
        w.forget_admit()
        self.assertFalse(w.admit())            # 17:28:45 the narration: refused

    def test_take_back_is_once_and_only_for_the_last_admit(self):
        c = _Clock(0.0)
        w = FollowupWindow(20, clock=c)
        self.assertFalse(w.take_back_admit())  # nothing admitted
        w.note_addressed()
        self.assertFalse(w.take_back_admit())  # a wake-word turn is not taken back
        c.t = 5.0
        self.assertTrue(w.admit())
        self.assertTrue(w.take_back_admit())
        self.assertFalse(w.take_back_admit())  # once
        self.assertAlmostEqual(w.remaining_s(), 15.0)

    def test_an_earlier_turns_admit_is_never_taken_back(self):
        c = _Clock(0.0)
        w = FollowupWindow(20, clock=c)
        w.note_addressed()
        c.t = 10.0
        self.assertTrue(w.admit())             # his own follow-up, kept
        c.t = 15.0
        w.forget_admit()                       # next turn: the gate did not ask admit()
        self.assertFalse(w.take_back_admit())
        self.assertAlmostEqual(w.remaining_s(), 15.0)

    def test_a_wake_word_after_an_admit_wins(self):
        c = _Clock(0.0)
        w = FollowupWindow(20, clock=c)
        w.note_addressed()
        c.t = 5.0
        w.admit()
        c.t = 6.0
        w.note_addressed()
        self.assertFalse(w.take_back_admit())
        self.assertAlmostEqual(w.remaining_s(), 20.0)

    def test_disabled_window_has_nothing_to_take_back(self):
        w = FollowupWindow(0, clock=_Clock())
        self.assertFalse(w.admit())
        self.assertFalse(w.take_back_admit())


class KeepsExtensionTests(unittest.TestCase):
    def test_media_time_unsure_or_not_owner_does_not_extend(self):
        self.assertFalse(mg.keeps_followup_extension(True, lg.UNSURE))
        self.assertFalse(mg.keeps_followup_extension(True, lg.NOT_OWNER))

    def test_everything_else_keeps_todays_behaviour(self):
        self.assertTrue(mg.keeps_followup_extension(True, lg.OWNER))
        self.assertTrue(mg.keeps_followup_extension(True, lg.UNAVAILABLE))
        for v in (lg.UNSURE, lg.NOT_OWNER, lg.OWNER, lg.UNAVAILABLE, None):
            self.assertTrue(mg.keeps_followup_extension(False, v))


if __name__ == "__main__":
    unittest.main()
