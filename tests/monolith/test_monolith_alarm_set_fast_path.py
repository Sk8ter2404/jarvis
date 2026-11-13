"""The set-alarm fast path: a plain ask for an alarm at a time is set by
the alarm skill with no brain call (bobert_companion._run_alarm_set_shortcut,
core/alarm_intent.py).

Live 2026-11-12: "set an alarm for 422" reached set_alarm through the brain
at 16:21:18 and was answered "422 what?" at 17:03:22.

The alarm skill is the real skills/alarm.py, loaded isolated, with arming,
saving and every machine hook replaced: no timer thread, no file, no sound.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import threading
import time as _time
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests._skill_harness import load_skill_isolated


@requires_monolith
class AlarmSetFastPathTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoken = []
        self._p(bc, "_speak", side_effect=lambda t, *a, **k: self.spoken.append(t))
        self.set_state = self._p(bc, "set_state")
        self._p(bc, "FAST_PATHS_ENABLED", True)
        self._p(bc, "ALARM_SET_FAST_PATH", True, create=True)
        self.llm = self._p(bc, "_call_llm", side_effect=AssertionError("LLM called"))
        self._p(bc, "get_response_with_animation",
                side_effect=AssertionError("LLM called"))
        self.arm_filler = self._p(bc._processing_filler, "arm")
        self._p(bc, "maybe_replay_last_action", return_value=None)
        router = types.ModuleType("core.mode_router")
        router.maybe_handle_mode_toggle = lambda _t: None
        router.controlled_dispatch = lambda _t, _a: None
        router.is_in_controlled_mode = lambda: False
        disp = types.ModuleType("core.dispatcher")
        disp.resolve_and_dispatch = lambda _t, _a: None
        voice = types.SimpleNamespace(maybe_switch_backend=lambda _t: None)
        with contextlib.redirect_stdout(io.StringIO()):
            self.alarm, _acts = load_skill_isolated("alarm", register=False)
        self.armed = []

        def arm(secs, label, at=None, tries=0):
            self.armed.append((int(secs), label))
            return len(self.armed)

        for name, fn in (
                ("_arm", arm), ("_persist", lambda: None),
                ("_say", lambda *a, **k: None),
                ("_sound_start", lambda *a: None), ("_sound_stop", lambda: None),
                ("_beep", lambda: None), ("_com_enter", lambda: None),
                ("_to_speakers", lambda: None), ("_volume_read", lambda: None),
                ("_volume_write", lambda *a: True)):
            self._p(self.alarm, name, fn)
        # Thu 2026-11-12 16:21:00 for the skill's own clock reads.
        fixed = _dt.datetime(2026, 11, 12, 16, 21, 0).timestamp()
        self._p(self.alarm, "time", types.SimpleNamespace(
            time=lambda: fixed, monotonic=_time.monotonic, sleep=_time.sleep))
        patcher = mock.patch.dict(bc.sys.modules, {
            "core.mode_router": router, "core.dispatcher": disp,
            "skill_custom_voice": voice, "skill_alarm": self.alarm})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.set_calls = []

        def set_alarm(arg=""):
            self.set_calls.append(arg)
            return self.alarm.set_alarm(arg)

        actions = dict(bc.ACTIONS)
        actions["set_alarm"] = set_alarm
        actions["list_alarms"] = self.alarm.list_alarms
        self._p(bc, "ACTIONS", actions)
        self._p(bc, "conversation_history", [])
        self.session_action = self._p(bc, "record_session_action")
        self.action_history = self._p(bc, "record_action_history")
        # No alarm question is open unless a test opens one.
        self._p(bc, "_last_turn_ctx_at", [0.0])

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _run(self, text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.bc._run_voice_shortcuts(text)
        self.log = buf.getvalue()
        return out

    def _ring(self):
        self.alarm._ringing[0] = {"id": 9, "label": "", "at": None,
                                  "stop": threading.Event(), "why": ""}
        self.addCleanup(self.alarm._ringing.__setitem__, 0, None)
        return self.alarm._ringing[0]

    # -- it sets -----------------------------------------------------------
    def test_the_live_typed_line_sets_the_alarm_with_no_brain(self):
        self.assertTrue(self._run("Jarvis, set an alarm for 422"))
        self.assertEqual(self.set_calls, ["at 4:22"])
        self.assertEqual(len(self.armed), 1)
        self.assertEqual(len(self.spoken), 1)
        self.assertTrue(self.spoken[0].startswith(
            "Alarm #1 set for 4:22 PM, 60 seconds from now, sir."), self.spoken)
        self.assertIn("[fast-path] alarm-set", self.log)
        self.assertIn("JARVIS: Alarm #1 set for 4:22 PM", self.log)
        self.llm.assert_not_called()
        self.arm_filler.assert_not_called()
        # The trace a routed action leaves ("do that again", "what did you
        # just do").
        self.session_action.assert_called_once_with("set_alarm", "at 4:22")
        self.action_history.assert_called_once_with(
            "set_alarm", "at 4:22", self.spoken[0])

    def test_the_live_spoken_line_with_a_remark_after_it(self):
        self.assertTrue(self._run("Jarvis setting an alarm for 422. That's really good."))
        self.assertEqual(self.set_calls, ["at 4:22"])

    def test_wake_me_and_a_duration(self):
        self.assertTrue(self._run("wake me up at 6 30"))
        self.assertTrue(self._run("alarm in 20 minutes"))
        self.assertEqual(self.set_calls, ["at 6:30", "in 20 minutes"])
        self.assertEqual(self.armed[1][0], 1200)

    def test_the_turn_is_recorded_like_any_other(self):
        self._run("Jarvis, wake me up at 7 am")
        hist = self.bc.conversation_history
        self.assertEqual([m["role"] for m in hist], ["user", "assistant"])
        self.assertEqual(hist[0]["content"], "Jarvis, wake me up at 7 am")
        self.assertEqual(hist[1]["content"], self.spoken[0])
        self.assertIn("7:00 AM tomorrow", self.spoken[0])
        self.assertEqual(self.bc._last_user_text[0], "Jarvis, wake me up at 7 am")

    # -- it stands aside ---------------------------------------------------
    def test_lines_that_are_not_a_plain_ask_go_to_the_brain(self):
        for line in ("did you set an alarm", "what alarms should I set",
                     "don't set an alarm for 7", "cancel the alarm",
                     "snooze the alarm for 10 minutes", "my alarm went off at 7",
                     "set the smoke alarm for 7", "set a timer for 20 minutes",
                     "remind me at 7 to call mom", "tell mom to set an alarm for 7",
                     "set an alarm for 7 and turn off the lights",
                     "set an alarm for 7. Actually never mind.",
                     "set an alarm for 7. Just kidding."):
            with self.subTest(line=line):
                self.assertFalse(self.bc._run_alarm_set_shortcut(line))
        self.assertEqual(self.set_calls, [])
        self.assertEqual(self.spoken, [])

    def test_no_time_goes_to_the_brain(self):
        for line in ("set an alarm", "wake me up", "wake me up in the morning",
                     "set an alarm for tomorrow", "set an alarm for 20"):
            with self.subTest(line=line):
                self.assertFalse(self.bc._run_alarm_set_shortcut(line))
        self.assertEqual(self.set_calls, [])

    def test_a_time_the_skill_cannot_read_goes_to_the_brain(self):
        self._p(self.alarm, "_parse_when", lambda _t: None)
        self.assertFalse(self._run("set an alarm for 7 am"))
        self.assertEqual(self.set_calls, [])
        self.assertEqual(self.spoken, [])

    def test_no_alarm_skill_or_a_switch_off_goes_to_the_brain(self):
        with mock.patch.dict(self.bc.sys.modules):
            self.bc.sys.modules.pop("skill_alarm")
            self.assertFalse(self.bc._run_alarm_set_shortcut("set an alarm for 7 am"))
        with mock.patch.object(self.bc, "ALARM_SET_FAST_PATH", False):
            self.assertFalse(self.bc._run_alarm_set_shortcut("set an alarm for 7 am"))
        self._p(self.bc, "FAST_PATHS_ENABLED", False)
        self.assertFalse(self.bc._run_alarm_set_shortcut("set an alarm for 7 am"))
        self.assertEqual(self.set_calls, [])

    def test_a_failing_skill_is_never_silent(self):
        self.bc.ACTIONS["set_alarm"] = lambda arg="": (_ for _ in ()).throw(OSError("x"))
        self.assertTrue(self._run("set an alarm for 7 am"))
        self.assertEqual(self.spoken, ["I could not set that alarm, sir. Nothing is set."])

    # -- a bare time that answers his own question -------------------------
    def _asked(self, ask="Jarvis, set an alarm for 422",
               reply="I'm afraid I'll need a bit more clarity, sir - 422 what?"):
        self.bc.conversation_history[:] = [
            {"role": "user", "content": ask},
            {"role": "assistant", "content": reply}]
        self.bc._last_turn_ctx_at[0] = _time.monotonic()

    def test_a_bare_time_answers_the_question_about_the_alarm(self):
        for said, arg in (("4:22", "4:22"), ("7 am", "7 am"), ("Jarvis, 7 a.m.", "7 a.m"),
                          ("at 7", "at 7"), ("7", "at 7"), ("seven", "at seven"),
                          ("about 7", "at 7"), ("the 7", "at 7"),
                          ("I mean 7", "at 7"), ("422", "422"),
                          ("5.30 in the morning", "5.30 in the morning"),
                          ("seven thirty", "seven thirty"),
                          ("in 20 minutes", "in 20 minutes")):
            with self.subTest(said=said):
                self._asked()
                self.set_calls.clear()
                self.assertTrue(self._run(said))
                self.assertEqual(self.set_calls, [arg])
        self.llm.assert_not_called()

    def test_a_bare_time_with_no_alarm_question_before_it_is_not_an_alarm(self):
        cases = (
            [],                                                 # nothing asked
            [{"role": "user", "content": "what is 5 plus 2"},
             {"role": "assistant", "content": "Which one, sir?"}],
            [{"role": "user", "content": "set an alarm for 7 am"},
             {"role": "assistant", "content": "Alarm #1 set for 7:00 AM, sir."}],
            [{"role": "user", "content": "set an alarm"},
             {"role": "assistant", "content": "Certainly, sir. [ACTION: set_alarm, 7]"}],
        )
        for hist in cases:
            with self.subTest(hist=hist):
                self.bc.conversation_history[:] = list(hist)
                self.bc._last_turn_ctx_at[0] = _time.monotonic()
                self.assertFalse(self.bc._run_alarm_set_shortcut("7 am"))
        # A question about an alarm that EXISTS is not a question about when
        # to set one: "the 7" must not become a new alarm.
        for ask, reply, said in (
                ("cancel my alarm", "Which one, sir - the 7 or the 8?", "the 7"),
                ("cancel the alarm", "Which alarm, sir?", "7 am"),
                ("snooze the alarm", "How many minutes, sir?", "10 minutes"),
                ("my alarm didn't go off this morning",
                 "Shall I check when it was set, sir?", "5 minutes"),
                ("set an alarm for 7 am", "Would you like it loud?", "9")):
            with self.subTest(ask=ask):
                self._asked(ask, reply)
                self.assertFalse(self.bc._run_alarm_set_shortcut(said))
        # An old question is not open any more.
        self._asked()
        self.bc._last_turn_ctx_at[0] = _time.monotonic() - 600
        self.assertFalse(self.bc._run_alarm_set_shortcut("7 am"))
        # After the question, anything that is not just a time is the brain's.
        for said in ("never mind", "7 dollars", "I have 2 dogs", "no",
                     "the one at 7 that we talked about earlier today please"):
            with self.subTest(said=said):
                self._asked()
                self.assertFalse(self.bc._run_alarm_set_shortcut(said))
        self.assertEqual(self.set_calls, [])

    def test_the_skills_own_question_opens_it_whatever_the_history_says(self):
        # The brain path does not write the skill's question into the
        # history, so the skill keeps the stamp.
        self.alarm.set_alarm("loud please")
        self.assertTrue(self.alarm.asked_when_recently())
        self.assertTrue(self._run("7 am"))
        self.assertEqual(self.set_calls, ["7 am"])
        self.assertFalse(self.alarm.asked_when_recently())

    def test_the_speech_filter_does_not_eat_the_answer(self):
        # "7 am" is four characters: is_valid_speech calls it too short, and
        # the main loop dropped it before any shortcut ran.
        conf = {"no_speech_prob": 0.05, "avg_logprob": -0.2}
        for said in ("422", "7 am", "at 7", "7"):
            with self.subTest(said=said):
                valid, _why = self.bc.is_valid_speech(said, conf, peak_rms=0.2)
                self.assertFalse(valid, "the filter changed: this test is stale")
                self._asked()
                self.assertTrue(self.bc._alarm_when_answer(said))
        self.bc.conversation_history[:] = []
        self.assertEqual(self.bc._alarm_when_answer("7 am"), "")

    # -- "what alarms do I have" -------------------------------------------
    def test_the_alarm_list_is_read_from_the_store(self):
        self.assertTrue(self._run("Jarvis, what alarms do I have?"))
        self.assertEqual(self.spoken, ["No alarms are set, sir."])
        self.assertIn("[fast-path] alarms", self.log)
        self.llm.assert_not_called()

    # -- the ringing shortcut keeps first place ----------------------------
    def test_a_ringing_alarm_is_answered_first(self):
        self._ring()
        self.assertTrue(self._run("stop"))
        self.assertEqual(self.spoken, ["Alarm off, sir."])
        self.assertEqual(self.set_calls, [])
        # A set line is not an answer: "to go off" is not "off", and the
        # ring goes on.
        me = self._ring()
        self.assertTrue(self._run("set an alarm to go off at 7 am"))
        self.assertEqual(self.set_calls, ["7:00 am"])
        self.assertFalse(me["stop"].is_set())

    def test_no_set_line_is_a_ringing_command_when_nothing_rings(self):
        for line in ("set an alarm for 7 am", "wake me up at 6 30",
                     "set an alarm to go off at 6 am", "alarm in 20 minutes"):
            self.assertEqual(self.bc._alarm_ringing_command(line), "", line)

    def test_standby_answers_a_ringing_alarm_and_stays_in_standby(self):
        self._ring()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(self.bc._alarm_ringing_shortcut("I'm up", keep_state=True))
        self.assertEqual(self.spoken, ["Alarm off, sir."])
        self.set_state.assert_not_called()
        self.assertTrue(self.bc._alarm_is_ringing())
        self.alarm._ringing[0] = None
        self.assertFalse(self.bc._alarm_is_ringing())

    def test_a_label_with_a_money_word_does_not_wait_on_a_yes(self):
        # "7 am | pay rent" used to be held for a spoken yes that wake-word
        # mode drops, like set_timer before it.
        self.assertFalse(self.bc._needs_confirmation("set_alarm", "7 am | pay rent"))
        self.assertFalse(self.bc._needs_confirmation("set_alarm", "6 am | buy tickets"))


if __name__ == "__main__":
    unittest.main()
