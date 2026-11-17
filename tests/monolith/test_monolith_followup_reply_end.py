"""FOLLOWUP_WINDOW_FROM_REPLY_END (3.0.26): the wake-word follow-up window
runs from the END of JARVIS's reply, through the live main-loop entry points.

Live 11-16..11-17 (wake-word mode, FOLLOWUP_WINDOW_S 20): the window opened
at the background gate (note_addressed / admit), about 0.3 s after he
stopped talking and before the brain, the actions and the reply. The median
voice turn ended 9.8 s later, and five turns ended 27-57 s later - the window
had closed before JARVIS stopped speaking, so his answer was dropped as
"wake-word mode - ignoring non-wake utterance".

Each turn here runs the main loop's gates in order (tests.monolith.
test_monolith_followup_unsure._FollowupBase._turn), an answered turn is
accepted the way main() accepts one (_note_owner_turn after "You:"), the
clock moves on by the reply's length, and the next pass's REAL
_note_turn_boundary re-arms the window. Fakes only (voice-ID, clock): no
audio device. The lines are neutral stand-ins, not the log's words.

    python tools/run_tests.py monolith.test_monolith_followup_reply_end
"""
from __future__ import annotations

import inspect
from unittest import mock

from tests.monolith.test_monolith_followup_unsure import _FollowupBase


class _ReplyEndBase(_FollowupBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        for cell in ("_turn_in_progress", "_utterance_in_progress",
                     "_last_owner_turn_at", "_prev_owner_turn_at",
                     "_last_convo_activity", "_music_capture_streak",
                     "_followup_cue_spoken"):
            c = getattr(bc, cell)
            self.addCleanup(c.__setitem__, 0, c[0])
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc._followup_cue_spoken[0] = False
        # The minute line reads devices and meters; not what this tests.
        p = mock.patch.object(bc, "_listen_tick")
        p.start()
        self.addCleanup(p.stop)
        # The shipped default (the live install does not set it).
        self.assertTrue(bc.FOLLOWUP_WINDOW_FROM_REPLY_END)

    def _boundary(self):
        """The next main-loop pass's top: the real _note_turn_boundary."""
        with mock.patch("builtins.print",
                        side_effect=lambda *a, **k: self.log.append(
                            " ".join(map(str, a)))):
            self.bc._note_turn_boundary()

    def _say(self, text, at, reply_s, **kw):
        """One mic turn at ``at``; when it is answered, JARVIS's reply ends
        ``reply_s`` later and the next pass begins."""
        got = self._turn(text, at, **kw)
        if got == "answered":
            self.bc._note_owner_turn()          # main(): after "You:"
            self.clock.t = at + reply_s
        self._boundary()
        return got


class ReplyEndTests(_ReplyEndBase):
    def test_his_answer_after_a_long_reply_is_admitted(self):
        # A 28 s reply (the 16:02 / 16:13 / 23:49 turns ran 27-29 s).
        self.assertEqual(self._say("Jarvis, explain the error.", 0, 28,
                                   name="owner", score=0.80), "answered")
        self.assertEqual(self._turn("What does that mean?", 33, name="owner",
                                    score=0.80), "answered")
        joined = "\n".join(self.log)
        self.assertIn("[follow-up] window re-armed after the reply", joined)
        self.assertIn("it had closed while JARVIS spoke", joined)
        self.assertNotIn("error", joined)        # numbers only

    def test_with_the_setting_off_the_answer_is_refused(self):
        with mock.patch.object(self.bc, "FOLLOWUP_WINDOW_FROM_REPLY_END", False):
            self._say("Jarvis, explain the error.", 0, 28, name="owner",
                      score=0.80)
            self.assertEqual(self._turn("What does that mean?", 33,
                                        name="owner", score=0.80), "refused")
        self.assertNotIn("re-armed", "\n".join(self.log))

    def test_a_typical_reply_leaves_the_whole_window(self):
        self._say("Jarvis, what's on today?", 0, 10, name="owner", score=0.80)
        self.assertAlmostEqual(self.win.remaining_s(), 20.0)
        self.assertIn("(10.0 s were left, now 20 s)", "\n".join(self.log))

    def test_a_kept_follow_up_re_arms_from_its_own_reply(self):
        self._say("Jarvis, what's on today?", 0, 8, name="owner", score=0.80)
        self.assertEqual(self._say("And tomorrow?", 15, 12, name="owner",
                                   score=0.80), "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 20.0)   # from t=27

    def test_an_unsure_follow_up_runs_but_does_not_re_arm(self):
        self._say("Jarvis, what's on today?", 0, 8, name="owner", score=0.80)
        # until 28; an unsure command at 15 runs, its extension taken back
        self.assertEqual(self._say("Close every window except the editor.",
                                   15, 10, score=0.61), "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 3.0)    # 28 - 25

    def test_an_unanswered_line_neither_re_arms_nor_blocks_the_take_back(self):
        self._say("Jarvis, what's on today?", 0, 8, name="owner", score=0.80)
        self.assertEqual(self._say("mm", 12, 0, noise=True), "noise")
        self.assertAlmostEqual(self.win._until, 32.0)          # admit pushed it
        # The next gate takes the unanswered extension back (3.0.7).
        self.clock.t = 20.0
        self.assertEqual(self._turn("The shelf is crooked.", 20, score=0.63),
                         "dropped")
        self.assertIn("the last admitted line was not answered",
                      "\n".join(self.log))

    def _wake_line_dropped_at_14(self, cue: bool):
        bc = self.bc
        self.clock.t = 14.0
        self.assertEqual(bc._bg_gate_for_turn("Jarvis, pause it.", False),
                         (False, ""))           # opens the window: until 34
        if cue:
            bc._followup_keep_for_reply()       # a gate dropped it, asked
        self.clock.t = 25.0                     # (the cue has been played)
        self._boundary()

    def test_a_spoken_drop_cue_re_arms_for_the_repeat(self):
        self._wake_line_dropped_at_14(cue=True)
        self.assertAlmostEqual(self.win.remaining_s(), 20.0)

    def test_a_silent_drop_does_not_re_arm(self):
        self._wake_line_dropped_at_14(cue=False)
        self.assertAlmostEqual(self.win.remaining_s(), 9.0)    # 34 - 25

    def test_a_typed_turn_never_re_arms(self):
        self._say("Jarvis, what's on today?", 0, 8, name="owner", score=0.80)
        self.assertEqual(self._say("what about tomorrow", 10, 15,
                                   injected=True), "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 3.0)    # 28 - 25

    def test_the_turn_boundary_runs_it_first(self):
        src = inspect.getsource(self.bc._note_turn_boundary)
        at = src.index("_followup_turn_over(bool(_turn_in_progress[0]))")
        self.assertLess(at, src.index("_turn_in_progress[0] = False"))

    def test_the_standby_greeting_counts_as_the_reply(self):
        src = inspect.getsource(self.bc._handle_sleep_standby)
        at = src.index("_speak(_greeting, volume_scale=_vol)")
        self.assertLess(at, src.index("_followup_note_cue()", at))


if __name__ == "__main__":
    import unittest
    unittest.main()
