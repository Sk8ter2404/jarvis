"""Tests for core/learn_gate.py — who may teach JARVIS (owner-only learning).

THE LIVE FAILURE (2026-09-30): with no wake word required, JARVIS answered a
phone call in the room and, because every ANSWERED turn counted as the
owner's, learned ten "facts about the user" from another person in one
afternoon. These pin the rule that stops it: a turn teaches only when it was
typed, led by the wake word, in the owner's enrolled voice, or a follow-up
inside a conversation one of those opened — and a voice that is confidently
someone else never teaches.

Pure logic, explicit timestamps: no I/O, no clock, no audio.

    python -m unittest tests.test_learn_gate
"""
from __future__ import annotations

import unittest

from core import learn_gate as lg


class VoiceVerdictTests(unittest.TestCase):
    def test_nobody_enrolled_is_no_evidence(self):
        self.assertEqual(lg.voice_verdict("someone", 0.9, enrolled=False),
                         lg.UNAVAILABLE)

    def test_a_match_with_memory_write_is_the_owner(self):
        self.assertEqual(lg.voice_verdict("owner", 0.81, enrolled=True),
                         lg.OWNER)

    def test_an_enrolled_guest_without_memory_write_is_not_the_owner(self):
        self.assertEqual(lg.voice_verdict("guest", 0.85, enrolled=True,
                                          may_write=False), lg.NOT_OWNER)

    def test_below_the_reject_floor_is_someone_else(self):
        self.assertEqual(lg.voice_verdict(None, 0.41, enrolled=True,
                                          reject_below=0.60), lg.NOT_OWNER)

    def test_between_the_floor_and_the_match_is_unsure(self):
        self.assertEqual(lg.voice_verdict(None, 0.66, enrolled=True,
                                          reject_below=0.60), lg.UNSURE)

    def test_no_embedding_is_no_evidence_not_a_stranger(self):
        for score in (0.0, -0.2, None, "x", float("nan")):
            with self.subTest(score=score):
                self.assertEqual(lg.voice_verdict(None, score, enrolled=True),
                                 lg.UNAVAILABLE)

    def test_a_malformed_floor_falls_back_to_the_default(self):
        self.assertEqual(lg.voice_verdict(None, 0.5, enrolled=True,
                                          reject_below="junk"), lg.NOT_OWNER)


class DecideTests(unittest.TestCase):
    def setUp(self):
        self.gate = lg.LearnGate(window_s=90)

    def test_the_live_failure_unaddressed_room_speech_does_not_teach(self):
        for voice in (lg.UNAVAILABLE, lg.UNSURE, lg.NOT_OWNER):
            with self.subTest(voice=voice):
                ok, why = lg.LearnGate(90).decide(100.0, voice=voice)
                self.assertFalse(ok)
                self.assertTrue(why)

    def test_typed_turns_always_teach(self):
        ok, why = self.gate.decide(10.0, injected=True, voice=lg.NOT_OWNER)
        self.assertTrue(ok)
        self.assertEqual(why, "typed")

    def test_wake_word_teaches_unless_the_voice_is_someone_else(self):
        self.assertTrue(self.gate.decide(10.0, wake=True)[0])
        self.assertTrue(self.gate.decide(20.0, wake=True, voice=lg.UNSURE)[0])
        self.assertFalse(lg.LearnGate(90).decide(10.0, wake=True,
                                                 voice=lg.NOT_OWNER)[0])

    def test_owner_voice_teaches_without_the_wake_word(self):
        self.assertTrue(self.gate.decide(10.0, voice=lg.OWNER)[0])

    def test_follow_ups_inside_the_window_teach(self):
        self.gate.decide(100.0, wake=True)
        self.assertTrue(self.gate.decide(150.0)[0])
        self.assertTrue(self.gate.decide(189.0, voice=lg.UNSURE)[0])

    def test_someone_else_inside_the_window_still_does_not_teach(self):
        self.gate.decide(100.0, wake=True)
        self.assertFalse(self.gate.decide(120.0, voice=lg.NOT_OWNER)[0])

    def test_window_lapses_after_the_last_positive_turn(self):
        self.gate.decide(100.0, voice=lg.OWNER)
        self.assertFalse(self.gate.decide(191.0)[0])

    def test_window_admitted_turns_never_extend_it(self):
        # followup_window.py's known risk: crosstalk holding the window open.
        self.gate.decide(100.0, wake=True)
        for ts in (150.0, 180.0):
            self.assertTrue(self.gate.decide(ts)[0])
        self.assertFalse(self.gate.decide(195.0)[0])

    def test_positive_turns_do_extend_it(self):
        self.gate.decide(100.0, wake=True)
        self.gate.decide(170.0, voice=lg.OWNER)
        self.assertTrue(self.gate.decide(250.0)[0])

    def test_a_standby_wake_opens_the_window(self):
        self.gate.note_wake(100.0)
        self.assertTrue(self.gate.decide(105.0)[0])
        self.assertFalse(self.gate.decide(300.0)[0])

    def test_a_turn_from_before_the_window_opened_is_outside_it(self):
        self.gate.note_wake(100.0)
        self.assertFalse(self.gate.decide(95.0)[0])

    def test_a_late_classified_older_turn_does_not_rewind_the_window(self):
        self.gate.decide(200.0, wake=True)
        self.gate.decide(150.0, injected=True)
        self.assertTrue(self.gate.decide(280.0)[0])

    def test_window_zero_means_only_clearly_owner_turns(self):
        gate = lg.LearnGate(window_s=0)
        gate.decide(100.0, wake=True)
        self.assertFalse(gate.decide(101.0)[0])

    def test_a_malformed_window_disables_never_widens(self):
        for bad in ("junk", None, float("nan"), float("inf"), -5):
            with self.subTest(window=bad):
                gate = lg.LearnGate(window_s=bad)
                self.assertEqual(gate.window_s, 0.0)

    def test_overheard_speech_needs_the_owner_voice_and_opens_nothing(self):
        ok, _ = self.gate.decide(100.0, voice=lg.OWNER, overheard=True)
        self.assertTrue(ok)
        self.assertFalse(self.gate.decide(110.0)[0])
        for voice in (lg.UNAVAILABLE, lg.UNSURE, lg.NOT_OWNER):
            with self.subTest(voice=voice):
                self.assertFalse(self.gate.decide(120.0, voice=voice,
                                                  overheard=True)[0])

    def test_overheard_speech_inside_the_window_still_needs_the_voice(self):
        self.gate.decide(100.0, wake=True)
        self.assertFalse(self.gate.decide(110.0, overheard=True)[0])

    def test_reasons_never_carry_turn_text(self):
        # decide() never sees the text at all; its reasons are fixed phrases.
        reasons = {self.gate.decide(t, voice=v)[1]
                   for t, v in ((1.0, lg.NOT_OWNER), (2.0, lg.UNAVAILABLE))}
        self.assertTrue(all(len(r) < 60 for r in reasons))


if __name__ == "__main__":
    unittest.main()
