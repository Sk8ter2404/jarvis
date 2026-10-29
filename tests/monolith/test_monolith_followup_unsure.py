"""FOLLOWUP_UNSURE_NO_EXTEND + FOLLOWUP_UNSURE_COMMANDS_ONLY (3.0.7): any
unsure-voice follow-up inside the wake-word follow-up window.

Live 2026-10-28 19:07-19:12 on 3.0.5 (wake-word mode, FOLLOWUP_WINDOW_S 20,
the PC audio MUTED from 19:09:15, so the 3.0.6 media rule could not apply):
room speech scoring 0.61-0.67 against the owner's voiceprint was admitted as
a follow-up, answered, and every admitted line extended the window - and so
did four low-confidence captures (19:09:28-19:09:39) that the speech filter
dropped AFTER the window had admitted them. The owner said "You're supposed
to be quiet." His own first follow-up that session ("close every window
except ...") scored 0.66, so a voice score cannot refuse a follow-up.

Through the live entry points in the main loop's order: _bg_gate_for_turn
(the background gate: admit() extends the window), _media_voice_gate, then
_followup_unsure_gate (the new step, just before the turn is used). A
capture the speech filter drops stops after the bg gate, as in main().
Fakes only (the media gate harness): no audio device, voiceprint or COM.
The lines are neutral stand-ins of the same shape, not the log's words.

    python -m unittest tests.monolith.test_monolith_followup_unsure
"""
from __future__ import annotations

import inspect
from unittest import mock

from core.followup_window import FollowupWindow
from tests.monolith.test_monolith_media_gate import _Base, _FakeVoiceId

OWNER = ("owner", None)     # (name, score): voice_id named him
UNSURE = (None, None)       # under voice_id's match threshold


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class _FollowupBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.clock = _Clock(0.0)
        self.win = FollowupWindow(20, clock=self.clock)
        self.room_talk = mock.Mock()
        for p in (mock.patch.object(bc, "_require_wake_runtime", True),
                  mock.patch.object(bc, "_followup_window", self.win),
                  mock.patch.object(bc, "_room_tv_gate", return_value=False),
                  mock.patch.object(bc, "_reply_prompt_pending", return_value=False),
                  mock.patch.object(bc, "_last_jarvis_line", [0.0, False]),
                  mock.patch.object(bc, "_capture_voice_memo", [None]),
                  mock.patch.object(bc, "_note_room_talk", self.room_talk),
                  mock.patch.object(bc, "FOLLOWUP_UNSURE_NO_EXTEND", True, create=True),
                  mock.patch.object(bc, "FOLLOWUP_UNSURE_COMMANDS_ONLY", True,
                                    create=True)):
            p.start()
            self.addCleanup(p.stop)
        self.log = []

    def _turn(self, text, at, *, name=None, score=0.64, peak=0.0, noise=False,
              injected=False, unclear=None):
        """One mic turn at ``at`` s. 'refused' (bg gate), 'noise' (admitted,
        then the speech filter dropped it: nothing after the bg gate runs),
        'media' (media gate drop), 'dropped' (the new gate) or 'answered'.
        ``unclear`` = "ask" / "silent" runs the live _unclear_turn after the
        media gate: 'unclear-ask' (it spoke the cue) / 'unclear-silent'."""
        bc = self.bc
        self.clock.t = at
        bc._last_capture_audio = bc.np.zeros(16000 * 2, dtype="float32")  # a new capture
        vid = _FakeVoiceId(name, score)
        import core.voice_id as real_vid
        with mock.patch.object(bc, "_media_probe_result", return_value=peak,
                               create=True), \
                mock.patch.object(real_vid, "list_enrolled", side_effect=vid.list_enrolled), \
                mock.patch.object(real_vid, "is_available", side_effect=vid.is_available), \
                mock.patch.object(real_vid, "identify_speaker",
                                  side_effect=lambda a, sr: vid.identify_speaker(a, sr)), \
                mock.patch.object(real_vid, "can", side_effect=vid.can), \
                mock.patch("builtins.print",
                           side_effect=lambda *a, **k: self.log.append(" ".join(map(str, a)))):
            refuse, why = bc._bg_gate_for_turn(text, injected)
            if refuse:
                return "refused"
            if noise:
                return "noise"
            if bc._media_voice_gate(text, injected):
                return "media"
            if unclear:
                self.spoke = mock.Mock()
                owner_at = bc.time.monotonic() if unclear == "ask" else 0.0
                with mock.patch.object(bc._speech_filter_mod, "unclear_reason",
                                       return_value="marginal level"),                         mock.patch.object(bc, "_last_owner_turn_at", [owner_at]),                         mock.patch.object(bc, "_unclear_cue_at", [0.0]),                         mock.patch.object(bc, "_speak", self.spoke):
                    if bc._unclear_turn(text, injected):
                        return "unclear-ask" if self.spoke.called else "unclear-silent"
            if bc._followup_unsure_gate(text, why, injected, {}):
                return "dropped"
        return "answered"


class LiveSequenceTests(_FollowupBase):
    """The 19:07-19:10 stretch, times in seconds from the first wake line."""

    def _replay(self):
        t = self._turn
        return [
            t("Jarvis, how are you feeling today?", 0, name="owner", score=0.65),
            t("Seems so.", 12, name="owner", score=0.76),
            t("Close every window except the editor, I'm heading out.", 25, score=0.66),
            t("Jarvis, stay quiet while I'm out.", 62, name="owner", score=0.73),
            t("That turned out really well, everyone.", 73, score=0.64),
            t("See you can't do it.", 81, score=0.67),
            t("I'm going to put the box on the top shelf.", 91, name="owner", score=0.90),
            t("Jarvis, mute all audio.", 129, name="owner", score=0.61),
            t("Nice work.", 136, name="owner", score=0.87),
            t("mm", 145, noise=True), t("hmm hmm", 147, noise=True),
            t("uh huh huh", 152, noise=True), t("the uh the um", 155, noise=True),
            t("The shelf is crooked.", 160, score=0.63),
            t("Should the river carry me home", 177, score=0.66),
        ]

    def test_room_lines_no_longer_hold_the_window_or_get_answers(self):
        got = self._replay()
        self.assertEqual(got, [
            "answered",   # wake
            "answered",   # his matched voice: extends to 32 s
            "answered",   # his first follow-up command at 0.66: STILL RUNS
            "answered",   # wake: re-opens to 82 s
            "dropped",    # unsure statement: unanswered, not extended
            "dropped",    # 81 < 82: admitted, unanswered
            "refused",    # 91 > 82: the window had closed (HEAD answered it)
            "answered",   # wake
            "answered",   # matched voice: extends to 156 s
            "noise", "noise", "noise", "noise",   # each taken back
            "refused",    # 160 > 156 (HEAD: admitted, answered)
            "refused",    # 177 (HEAD: admitted, answered)
        ])
        self.room_talk.assert_called()            # a dropped line is room talk
        joined = "\n".join(self.log)
        self.assertIn("[follow-up] window not extended: voice not confirmed", joined)
        self.assertIn("[follow-up] window not extended: the last admitted line "
                      "was not answered", joined)
        for words in ("shelf", "river", "editor", "everyone"):
            self.assertNotIn(words, joined)       # numbers only, never the words

    def test_his_unsure_command_runs_but_does_not_extend(self):
        self._turn("Jarvis, what's on today?", 0, name="owner", score=0.70)
        self.assertEqual(self._turn("Close every window except the editor.", 10,
                                    score=0.61), "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 10.0)   # not pushed to 20

    def test_both_off_is_the_old_behaviour(self):
        with mock.patch.object(self.bc, "FOLLOWUP_UNSURE_NO_EXTEND", False), \
                mock.patch.object(self.bc, "FOLLOWUP_UNSURE_COMMANDS_ONLY", False):
            got = self._replay()
        self.assertEqual(got.count("refused"), 0)          # held open throughout
        self.assertEqual(got.count("dropped"), 0)
        self.assertNotIn("[follow-up]", "\n".join(self.log))

    def test_commands_only_off_still_stops_the_extension(self):
        with mock.patch.object(self.bc, "FOLLOWUP_UNSURE_COMMANDS_ONLY", False):
            got = self._replay()
        self.assertEqual(got[4:7], ["answered", "answered", "refused"])
        self.assertEqual(got[-2:], ["refused", "refused"])
        self.assertEqual(got.count("dropped"), 0)


class GateTests(_FollowupBase):
    def _open(self):
        self._turn("Jarvis, are you there?", 0, name="owner", score=0.80)

    def test_an_unsure_question_is_answered(self):
        self._open()
        self.assertEqual(self._turn("Where is it?", 5, score=0.62), "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 15.0)

    def test_a_media_control_is_answered(self):
        self._open()
        self.assertEqual(self._turn("turn it down a bit", 5, score=0.62), "answered")

    def test_an_answer_to_his_question_is_answered(self):
        self._open()
        self.bc._last_jarvis_line[:] = [self.bc.time.monotonic(), True]
        self.assertEqual(self._turn("The left one.", 5, score=0.62), "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 15.0)   # still not extended

    def test_a_waiting_prompt_lets_a_bare_answer_through(self):
        self._open()
        with mock.patch.object(self.bc, "_reply_prompt_pending", return_value=True):
            self.assertEqual(self._turn("The left one.", 5, score=0.62), "answered")

    def test_a_not_owner_follow_up_does_not_extend(self):
        self._open()
        self.assertEqual(self._turn("What time is it?", 5, score=0.50), "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 15.0)

    def test_his_matched_voice_still_extends(self):
        self._open()
        self.assertEqual(self._turn("That went well.", 5, name="owner", score=0.85),
                         "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 20.0)

    def test_voice_id_unavailable_keeps_todays_behaviour(self):
        self._open()
        bc = self.bc
        bc._last_capture_audio = None      # no capture to score
        self.clock.t = 5
        with mock.patch("builtins.print"):
            refuse, why = bc._bg_gate_for_turn("That went well.", False)
            self.assertFalse(refuse)
            self.assertFalse(bc._followup_unsure_gate("That went well.", why, False))
        self.assertAlmostEqual(self.win.remaining_s(), 20.0)

    def test_typed_and_wake_led_turns_are_never_looked_at(self):
        self._open()
        self.assertEqual(self._turn("That went well.", 5, score=0.62, injected=True),
                         "answered")
        self.assertEqual(self._turn("Jarvis, that went well.", 6, score=0.62),
                         "answered")
        self.assertAlmostEqual(self.win.remaining_s(), 20.0)

    def test_the_bg_gate_reason_is_the_one_the_rule_returns(self):
        self._open()
        self.clock.t = 3
        with mock.patch("builtins.print"):
            self.assertEqual(self.bc._bg_gate_for_turn("and then", False),
                             (False, self.bc._BG_FOLLOWUP_WHY))

    def test_a_gate_error_fails_open(self):
        self._open()
        self.clock.t = 3
        with mock.patch("builtins.print"):
            refuse, why = self.bc._bg_gate_for_turn("that went well", False)
            with mock.patch.object(self.bc, "_learn_voice_verdict",
                                   side_effect=RuntimeError("x")):
                self.assertFalse(self.bc._followup_unsure_gate("that went well",
                                                               why, False))


class AskedToRepeatTests(_FollowupBase):
    """Review 3.0.7: JARVIS asked the owner to repeat, and the repeat was
    refused because the next turn took the window back."""

    def _ask_then_repeat(self, repeat_at=22, **kw):
        self._turn("Jarvis, set a timer for five minutes.", 0, name="owner", score=0.80)
        # admitted at 15 s (window -> 35 s), then the unclear gate speaks the cue
        first = self._turn("uh the", 15, unclear="ask")
        self.assertEqual(first, "unclear-ask")
        self.spoke.assert_called_once_with(self.bc.UNCLEAR_CUE)
        return self._turn("Make it ten minutes.", repeat_at, **kw)

    def test_the_repeat_after_a_cue_needs_no_wake_word(self):
        self.assertEqual(self._ask_then_repeat(22, name="owner", score=0.80), "answered")

    def test_the_repeat_in_an_unsure_voice_is_answered_too(self):
        self.assertEqual(self._ask_then_repeat(22, score=0.66), "answered")

    def test_the_cue_keeps_the_window_it_opened_not_more(self):
        # admitted at 15 s: open until 35 s; the repeat at 36 s needs the name
        self.assertEqual(self._ask_then_repeat(36, name="owner", score=0.80), "refused")

    def test_a_silent_unclear_drop_is_still_taken_back(self):
        self._turn("Jarvis, set a timer for five minutes.", 0, name="owner", score=0.80)
        self.assertEqual(self._turn("uh the", 15, unclear="silent"), "unclear-silent")
        self.spoke.assert_not_called()
        # window back to its 20 s deadline: 22 s is refused, as for noise
        self.assertEqual(self._turn("Make it ten minutes.", 22, name="owner", score=0.80),
                         "refused")

    def test_off_is_the_old_path(self):
        with mock.patch.object(self.bc, "FOLLOWUP_UNSURE_NO_EXTEND", False):
            self.assertEqual(self._ask_then_repeat(22, name="owner", score=0.80),
                             "answered")

    def test_a_media_drop_cue_after_a_wake_line_keeps_the_window(self):
        bc = self.bc
        self._turn("Jarvis, play something.", 0, name="owner", score=0.80)
        self.clock.t = 3
        speak = mock.Mock()
        with mock.patch.object(bc, "_speak", speak),                 mock.patch.object(bc, "_media_drop_cue_at", [0.0]),                 mock.patch("builtins.print"):
            self.assertEqual(bc._bg_gate_for_turn("Jarvis, find me a restaurant", False),
                             (False, ""))
            bc._media_gate_drop_cue("Jarvis, find me a restaurant")
        speak.assert_called_once_with(bc._media_gate.DROP_CUE)
        self.assertEqual(self._turn("Find me a restaurant.", 14, name="owner", score=0.80),
                         "answered")


class ContinuationTests(_FollowupBase):
    """Review 3.0.7: his short follow-ups that continue the last request,
    in his unsure range (0.61-0.72), were dropped without a word. Neutral
    stand-ins of the same shape."""

    LINES = ("And tomorrow.", "Same for the kitchen.", "Ten minutes instead.",
             "A bit quieter.", "Is it going to rain tomorrow.", "Are you there")

    def test_short_continuations_are_answered_at_0_66(self):
        for i, line in enumerate(self.LINES):
            base = 1000.0 * (i + 1)
            self.clock.t = base
            self.win.note_addressed()
            with self.subTest(line=line):
                self.assertEqual(self._turn(line, base + 5, score=0.66), "answered")
                # answered, but an unsure voice still does not extend
                self.assertAlmostEqual(self.win.remaining_s(), 15.0)

    def test_room_statements_are_still_dropped(self):
        for i, line in enumerate(("And then he said it was all fine, honestly.",
                                  "It was a bit quieter", "That was nice.",
                                  "The shelf is crooked.", "Should the river carry me home")):
            base = 1000.0 * (i + 1)
            self.clock.t = base
            self.win.note_addressed()
            with self.subTest(line=line):
                self.assertEqual(self._turn(line, base + 5, score=0.66), "dropped")

    def test_commands_only_off_changes_nothing_for_them(self):
        with mock.patch.object(self.bc, "FOLLOWUP_UNSURE_COMMANDS_ONLY", False):
            self.win.note_addressed()
            self.assertEqual(self._turn("And tomorrow.", 5, score=0.66), "answered")


class WiringTests(_FollowupBase):
    """Source-level: main() cannot run in a test."""

    def test_main_runs_the_gate_once_after_the_other_drops_and_before_use(self):
        src = inspect.getsource(self.bc.main)
        call = "if _followup_unsure_gate(text, _bg_why,"
        self.assertEqual(src.count("_followup_unsure_gate("), 1)
        at = src.index(call)
        for earlier in ("_bg_gate_for_turn(", "is_valid_speech(text, conf",
                        "if _media_voice_gate(text, _injected_text is not None):",
                        "if _unclear_turn(text, _injected_text is not None):"):
            self.assertLess(src.index(earlier), at, earlier)
        for later in ('print(f"  You:    {text}")',
                      "reply = _run_llm_dispatch(text",
                      "learn_from_turn(text, reply, memory"):
            self.assertLess(at, src.index(later), later)
        self.assertIn("continue", src[at:at + 200])

    def test_the_unclear_and_media_cues_keep_the_window_before_speaking(self):
        for fn in (self.bc._unclear_turn, self.bc._media_gate_drop_cue):
            src = inspect.getsource(fn)
            self.assertIn("_followup_keep_for_reply()", src, fn.__name__)
            self.assertLess(src.index("_followup_keep_for_reply()"),
                            src.index("_speak("), fn.__name__)

    def test_the_bg_gate_settles_the_previous_admit_first(self):
        src = inspect.getsource(self.bc._should_refuse_background_audio)
        self.assertLess(src.index("_followup_settle_previous()"),
                        src.index("_bg_refuse_rule("))

    def test_config_defaults(self):
        import core.config as cfg
        self.assertIs(cfg.FOLLOWUP_UNSURE_NO_EXTEND, True)
        self.assertIs(cfg.FOLLOWUP_UNSURE_COMMANDS_ONLY, True)
        self.assertIs(cfg.FOLLOWUP_MEDIA_NO_EXTEND, True)


if __name__ == "__main__":
    import unittest
    unittest.main()
