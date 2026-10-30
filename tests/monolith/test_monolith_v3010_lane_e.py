"""v3.0.10 lane E - conversation fixes from the 2026-10-28/29 live log.

  * MEDIA_TOGGLE_NO_FOLLOWUP: live 2026-10-29 18:53:21 "pause movie" ->
    pause_music answered "paused Chrome, sir", the prose was held and a
    second model round said "Understood, sir. I've paused Chrome, sir." (first
    audio 7.3 s). A pause / resume that plainly worked is not read back by a
    second round.
  * ACTION_TOPIC_GATE: live 18:55:45 the misheard "Much more percentage, so
    I've got" ran server_status. A read-only server / printer check whose
    subject his line never names is not run; JARVIS asks him to repeat.
  * RECALL_ACTION_GROUNDING: live 18:54:51, 14 s after the robot's speak
    action ran, "... sent it over to the robot?" got "a bit of a
    misinterpretation". The recall turn's prompt now lists what ran.
  * the turn-flag line carries the new flags.

Drives the REAL parse_and_run_actions / _run_llm_dispatch_body with stub
actions and a recording _speak (the answer-first harness). No audio, no LLM,
no media keys, no devices.

    python tools/run_tests.py monolith.test_monolith_v3010_lane_e
"""
from __future__ import annotations

import contextlib
import io
import time

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

_PAUSED = "paused Chrome, sir"


@requires_monolith
class _LaneEBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        self._p(bc, "_followup_keep_for_reply")
        self.paused = []
        bc.ACTIONS["pause_music"] = (
            lambda a="": self.paused.append(a) or _PAUSED)
        self.server_calls = []
        bc.ACTIONS["server_status"] = (
            lambda a="": self.server_calls.append(a)
            or "The server is up, sir.")

    def _run_turn(self, reply, text):
        """One owner turn: the grounding ledger open, as _run_llm_dispatch
        opens it around the body."""
        bc = self.bc
        prev = bc._begin_turn_grounding(text)
        try:
            return self._run(reply, text=text)
        finally:
            bc._end_turn_grounding(prev)


class MediaToggleNoFollowupTests(_LaneEBase):
    def test_pause_that_worked_speaks_the_prose_with_no_second_round(self):
        printed = self._run_turn(
            "Right away, sir. [ACTION: pause_music] I've paused the media, "
            "sir.", "freeze the film for a second please")
        self.assertEqual(self.paused, [""])
        self.followup.assert_not_called()
        self.assertIn("I've paused the media", " ".join(self.spoken))
        self.assertIn("[media-toggle] pause_music worked", printed)
        self.assertNotIn("Reading results", printed)

    def test_a_bare_token_speaks_the_result_line(self):
        self._run_turn("[ACTION: pause_music]", "pause the music")
        self.followup.assert_not_called()
        self.assertEqual(self.spoken, ["Paused Chrome, sir."])

    def test_already_paused_counts_as_done(self):
        self.bc.ACTIONS["pause_music"] = (
            lambda a="": "Chrome is already paused, sir")
        self._run_turn("Very good, sir. [ACTION: pause_music]",
                       "pause it")
        self.followup.assert_not_called()

    def test_nothing_playing_still_goes_back_to_the_model(self):
        self.bc.ACTIONS["pause_music"] = (
            lambda a="": "Nothing seems to be playing, sir - open Apple "
                         "Music and I'll take it from there.")
        self.followup.side_effect = ["Nothing is playing, sir.", ""]
        self._run_turn("[ACTION: pause_music] Paused, sir.", "pause it")
        self.followup.assert_called()
        self.assertNotIn("Paused, sir.", " ".join(self.spoken))

    def test_flag_off_is_the_old_hold_and_round(self):
        self._p(self.bc, "MEDIA_TOGGLE_NO_FOLLOWUP", False, create=True)
        self.followup.side_effect = ["I've paused Chrome, sir.", ""]
        printed = self._run_turn(
            "Right away, sir. [ACTION: pause_music] I've paused the media, "
            "sir.", "freeze the film for a second please")
        self.followup.assert_called()
        self.assertIn("[result-hold]", printed)
        self.assertNotIn("[media-toggle]", printed)

    def test_helper_reads_only_plain_successes(self):
        done = self.bc._media_toggle_done
        self.assertTrue(done("pause_music", "paused Chrome, sir"))
        self.assertTrue(done("resume_music", "resumed Chrome, sir"))
        self.assertTrue(done("resume_music",
                             "Chrome is already playing, sir"))
        self.assertFalse(done("pause_music", "media play/pause pressed"))
        self.assertFalse(done("pause_music",
                              "Chrome didn't accept that, sir."))
        self.assertFalse(done("next_song",
                              "skipped to the next track in Chrome, sir"))
        self.assertFalse(done("pause_music", None))


class ActionTopicGateTests(_LaneEBase):
    def test_live_misheard_line_does_not_run_server_status(self):
        printed = self._run_turn("[ACTION: server_status]",
                                 "Much more percentage, so I've got")
        self.assertEqual(self.server_calls, [])
        self.assertIn(self.bc.UNCLEAR_CUE, self.spoken)
        self.followup.assert_not_called()
        line = [ln for ln in printed.splitlines() if "[action-topic]" in ln]
        self.assertEqual(len(line), 1)
        self.assertNotIn("percentage", line[0])
        self.bc._followup_keep_for_reply.assert_called()

    def test_real_asks_run(self):
        for text in ("is the server up?", "Jarvis, how's unraid doing",
                     "any disks down?"):
            with self.subTest(text=text):
                self.server_calls.clear()
                self.spoken.clear()
                self._run_turn("[ACTION: server_status]", text)
                self.assertEqual(self.server_calls, [""])
                self.assertNotIn(self.bc.UNCLEAR_CUE, self.spoken)

    def test_a_follow_up_after_a_server_turn_runs(self):
        bc = self.bc
        bc.conversation_history.append(
            {"role": "user", "content": "is the server up?"})
        bc.conversation_history.append(
            {"role": "assistant", "content": "[ACTION: server_status]"})
        self._run_turn("[ACTION: server_status]", "how about now?")
        self.assertEqual(self.server_calls, [""])

    def test_flag_off_runs_it(self):
        self._p(self.bc, "ACTION_TOPIC_GATE", False, create=True)
        self._run_turn("[ACTION: server_status]",
                       "Much more percentage, so I've got")
        self.assertEqual(self.server_calls, [""])

    def test_control_actions_are_never_checked(self):
        self._run_turn("[ACTION: pause_music]", "Much more percentage")
        self.assertEqual(self.paused, [""])

    def test_no_owner_words_is_never_checked(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc.parse_and_run_actions("[ACTION: server_status]")
        self.assertEqual(self.server_calls, [""])


class RecallGroundingTests(_LaneEBase):
    def _history(self, entries):
        bc = self.bc
        self._p(bc, "_action_history", list(entries))

    def test_recall_question_gets_the_recent_actions(self):
        now = time.time()
        self._history([{"action": "robot_speak", "arg": "a rude line",
                        "result": "The robot said it, sir.", "at": now - 14}])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            note = self.bc._recall_action_note(
                "Wait, what? They sent it over to the robot?")
        self.assertIn("robot_speak", note)
        self.assertIn("The robot said it, sir.", note)
        self.assertIn("never deny", note)
        self.assertIn("[recall-grounding] 1 recent action(s)", buf.getvalue())
        self.assertNotIn("rude", buf.getvalue())

    def test_old_entry_other_line_or_flag_off_add_nothing(self):
        now = time.time()
        self._history([{"action": "robot_speak", "arg": "hi",
                        "result": "The robot said it, sir.",
                        "at": now - 120}])
        self.assertEqual(self.bc._recall_action_note("Wait, what?"), "")
        self._history([{"action": "robot_speak", "arg": "hi",
                        "result": "The robot said it, sir.", "at": now - 5}])
        self.assertEqual(self.bc._recall_action_note("tell me a joke"), "")
        self._p(self.bc, "RECALL_ACTION_GROUNDING", False, create=True)
        self.assertEqual(self.bc._recall_action_note("Wait, what?"), "")


class TurnFlagKeysTests(_LaneEBase):
    def test_new_flags_are_on_the_turn_flag_line(self):
        for key in ("VOICE_MOOD_SWEAR_ALONE_STRESSED",
                    "VOICE_MOOD_QUOTED_WORDS_IGNORED",
                    "VOICE_MOOD_CALMING_LIGHTS", "VOICE_MOOD_HONEST_ADDENDUM",
                    "MEDIA_TOGGLE_NO_FOLLOWUP", "ACTION_TOPIC_GATE",
                    "RECALL_ACTION_GROUNDING"):
            with self.subTest(key=key):
                self.assertIn(key, self.bc._TURN_FLAG_KEYS)
                self.assertTrue(hasattr(self.bc, key))
