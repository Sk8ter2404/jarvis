"""Answer first (2026-09-29): the model's short lead-in ("One moment, sir.")
is not spoken when the turn is about to speak a real answer.

Drives the REAL _run_llm_dispatch_body -> parse_and_run_actions ->
_speak_verbatim_results path with stub actions and a recording _speak. The
LLM call is replaced by a fake that honours _call_llm's history contract (the
full reply is appended to conversation_history), so the tests can also prove
the skipped lead-in is still remembered.

Covers:
  * verbatim action + short acknowledgement -> only the answer is spoken; the
    lead-in stays in conversation_history; [answer-first] log line carries no
    text;
  * the length / digit / question-mark rules keep a lead-in;
  * only an ACKNOWLEDGEMENT is ever skipped: a lead-in that carries content
    (a second answer, a refusal, a confirmation) is spoken;
  * every action in the reply must bring its own answer: a side-effect,
    fire-and-exit, failed or empty action keeps the lead-in;
  * a long verbatim answer keeps the lead-in unless the filler already spoke;
  * informative action: dropped only when the processing filler will really
    acknowledge this voice turn (armed here, not cancelled, clips cached);
  * leading [intent:] / [mood:] / [wry] tags are not lead-in words;
  * the decision is made before the random quip layer;
  * no action, pushback, CONFIRM_KEYWORDS, ambiguity, barge-in, preemptive
    hallucination injection and the early-streamed tail keep today's speech;
  * ANSWER_FIRST_ENABLED=False is identical to the old behaviour;
  * the [turn-timing] line carries lead_dropped=0/1.

No real audio, no LLM, no Ollama.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_answer_first
"""
from __future__ import annotations

import contextlib
import io
import threading
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

# Generic fixtures only.
_ANSWER = "Currently mild and clear, sir."            # digit-free verbatim result
_LEAD = "One moment, sir."
_LONG_LEAD = ("Right away sir, I will go and look into that for you now and "
              "report back as soon as I possibly can.")   # 21 words
# 15 words, every one an acknowledgement word.
_ACK_15 = ("Of course, sir, I will look into that for you right away, "
           "just one moment.")
_ACK_16 = ("Of course, sir, I will look into that for you right away now, "
           "just one moment.")
_LONG_ANSWER = " ".join(["Clear skies across the region this afternoon with a "
                         "light breeze from the west"] * 3) + ", sir."


class _NoThread:
    """ProcessingFiller thread factory that never starts a real thread."""

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self.target = target

    def start(self):
        pass


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._hist_len = len(bc.conversation_history)
        self.addCleanup(self._restore_hist)
        self.spoken: list[str] = []
        self._p(bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
        self._p(bc, "maybe_glance_response", return_value=None)
        self._p(bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_write_hud_state")
        self._p(bc, "record_session_action")
        self._p(bc, "record_action_history")
        self._p(bc, "PC_CONTROL_ENABLED", True)
        self._p(bc, "MID_TASK_STATUS_ENABLED", False)
        self._p(bc, "ANSWER_FIRST_ENABLED", True)
        self._p(bc, "_tts_interrupt_seq", [0])
        self._p(bc, "_stream_spoken_prefix", [""])
        self._p(bc, "_needs_confirmation", return_value=False)
        self._p(bc, "_jarvis_pushback", return_value=None)
        # No filler armed unless a test arms one (a typed turn). When one is
        # armed, its first-stage clips are cached and nothing suppresses it
        # (a warm cache on a normal voice turn) unless a test says otherwise.
        self._p(bc, "_processing_filler", self._filler())
        self._p(bc, "_filler_clips", self._clips(warm=True))
        self._p(bc, "_filler_suppressed", return_value=None)
        self.followup = self._p(bc, "get_followup_response", return_value="")
        self._p(bc, "ACTIONS", dict(bc.ACTIONS))
        bc.ACTIONS["weather_briefing"] = lambda a="": _ANSWER
        bc.ACTIONS["see_screen"] = lambda a="": "A text editor is open."
        bc.ACTIONS["volume_up"] = lambda a="": "Volume up."

    def _restore_hist(self):
        del self.bc.conversation_history[self._hist_len:]

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _filler(self, first=2.5):
        from core import processing_filler as pf
        return pf.ProcessingFiller(play_fn=lambda *a, **k: None,
                                   suppressed_fn=lambda: None,
                                   delays_fn=lambda: (first, 12.0),
                                   thread_factory=_NoThread)

    def _clips(self, warm: bool):
        from core import processing_filler as pf
        cache = pf.ClipCache(render_fn=lambda t: None, lock=threading.Lock(),
                             key_fn=lambda: "test-voice")
        if warm:
            for line in pf.FIRST_LINES + pf.STILL_LINES:
                self.assertTrue(cache.put(line, ([0.0] * 800, 16000)))
        return cache

    def _arm_filler(self):
        f = self.bc._processing_filler
        turn = f.arm()
        self.assertIsNotNone(turn)
        self.addCleanup(f.disarm, turn)
        return turn

    def _run(self, reply, text="what's the weather", during_llm=None):
        """One dispatch with a canned LLM reply. Returns the printed output."""
        bc = self.bc

        def fake_llm(user_text):
            # _call_llm's contract: user + the FULL reply go to history.
            bc.conversation_history.append({"role": "user",
                                            "content": user_text})
            bc.conversation_history.append({"role": "assistant",
                                            "content": reply})
            if during_llm is not None:
                during_llm()
            return reply

        self._p(bc, "get_response_with_animation", side_effect=fake_llm)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = bc._run_llm_dispatch_body(text)
        self.assertEqual(out, reply)
        return buf.getvalue()

    def _history_text(self):
        return " ".join(m.get("content", "") for m in
                        self.bc.conversation_history[self._hist_len:])


class VerbatimAnswerFirstTests(_Base):
    def test_short_lead_dropped_answer_spoken_history_kept(self):
        printed = self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])
        self.assertIn(_LEAD, self._history_text())
        self.assertIn("[answer-first] dropped lead-in (3 words)", printed)
        # The log line never carries the text itself.
        af = [ln for ln in printed.splitlines() if "[answer-first]" in ln]
        self.assertEqual(len(af), 1)
        self.assertNotIn("moment", af[0])

    def test_long_lead_is_spoken(self):
        self.assertGreater(len(_LONG_LEAD.split()), 15)
        printed = self._run(f"{_LONG_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LONG_LEAD, _ANSWER])
        self.assertNotIn("[answer-first]", printed)

    def test_fifteen_words_is_still_short(self):
        self.assertEqual(len(_ACK_15.split()), 15)
        self._run(f"{_ACK_15} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])

    def test_sixteen_words_is_spoken(self):
        self.assertEqual(len(_ACK_16.split()), 16)
        self._run(f"{_ACK_16} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ACK_16, _ANSWER])

    def test_lead_with_digits_is_spoken(self):
        lead = "It's 3 PM, sir."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [lead, _ANSWER])

    def test_lead_with_question_is_spoken(self):
        lead = "Shall I check the forecast too, sir?"
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [lead, _ANSWER])

    def test_failed_verbatim_result_keeps_lead(self):
        # A failed result is no answer, so ANSWER-FIRST keeps the lead-in.
        # Since 2026-09-30 a pure acknowledgement before a FAILED action is
        # not voiced anyway - by the other rule (tests/monolith/
        # test_monolith_ack_before_failure.py) - and the failure follow-up
        # speaks instead; answer-first itself still does not fire.
        bc = self.bc
        failed = "could not reach the weather service"
        bc.ACTIONS["weather_briefing"] = lambda a="": failed
        self.assertEqual(bc._answer_first_drop_count(
            f"{_LEAD} [ACTION: weather_briefing]", _LEAD,
            [("weather_briefing", failed, False)], "fired"), 0)
        self.followup.side_effect = ["The weather service is down, sir.", ""]
        printed = self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertNotIn("[answer-first]", printed)
        self.assertIn("[ack-hold]", printed)
        self.assertEqual(self.spoken, ["The weather service is down, sir."])

    def test_blank_verbatim_result_keeps_lead(self):
        # "A non-empty spoken result": whitespace is no answer.
        self.bc.ACTIONS["weather_briefing"] = lambda a="": "   "
        printed = self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LEAD])
        self.assertNotIn("[answer-first]", printed)

    def test_side_effect_action_only_keeps_lead(self):
        self._run(f"{_LEAD} [ACTION: volume_up]")
        self.assertEqual(self.spoken, [_LEAD])

    def test_inlined_answer_is_spoken_once(self):
        # The model already put the (digit-free) answer in its prose: it is
        # spoken exactly once.
        self._run(f"{_ANSWER} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])

    def test_disabled_is_identical_to_old_behaviour(self):
        self._p(self.bc, "ANSWER_FIRST_ENABLED", False)
        printed = self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LEAD, _ANSWER])
        self.assertNotIn("[answer-first]", printed)

    def test_setting_defaults_on_when_absent(self):
        # The consumer reads the setting with a True default: a monolith
        # without the global (an old user_settings / config) still answers
        # first.
        bc = self.bc
        with mock.patch.dict(bc.__dict__):
            del bc.__dict__["ANSWER_FIRST_ENABLED"]
            self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])

    def test_early_streamed_reply_tail_is_not_touched(self):
        # The streaming flush already voiced the first sentence; the tail is
        # the rest of the reply, not a lead-in.
        bc = self.bc
        head = "Checking the sky for you. "
        bc._stream_spoken_prefix[0] = head
        self._run(f"{head}{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LEAD, _ANSWER])


class AcknowledgementOnlyTests(_Base):
    """Only a lead-in that is pure acknowledgement is skipped."""

    def test_spec_example_lead_is_dropped(self):
        lead = "I'm afraid I'll have to check on that for you, sir."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])

    def test_checking_named_thing_is_dropped(self):
        lead = "Checking the weather for you now, sir."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])

    def test_compound_question_answer_is_spoken(self):
        # The owner asked two things; the prose answers one of them.
        lead = "Paris is the capital of France, sir."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [lead, _ANSWER])

    def test_refusal_is_spoken(self):
        lead = "I won't delete those, sir, but here is the status."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [lead, _ANSWER])

    def test_checking_with_a_finding_is_spoken(self):
        lead = "Checking the battery, it is charging, sir."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [lead, _ANSWER])

    def test_acknowledgement_then_content_is_spoken(self):
        lead = "One moment, sir. The kettle has boiled."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [lead, _ANSWER])


class EveryActionAnswersTests(_Base):
    """Every action in the reply must bring its own answer."""

    def test_side_effect_plus_verbatim_keeps_confirmation(self):
        lead = "Timer set for ten minutes, sir."
        self.bc.ACTIONS["set_timer"] = lambda a="": "Timer started."
        self._run(f"{lead} [ACTION: set_timer, 10 minutes] "
                  "[ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [lead, _ANSWER])

    def test_acknowledgement_plus_side_effect_plus_verbatim_is_spoken(self):
        self._run(f"{_LEAD} [ACTION: volume_up] [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LEAD, _ANSWER])

    def test_fire_and_exit_plus_verbatim_is_spoken(self):
        bc = self.bc
        name = sorted(bc._FIRE_AND_EXIT_ACTIONS)[0]
        bc.ACTIONS[name] = lambda a="": "scheduled"
        lead = "Right away, sir."
        self._run(f"{lead} [ACTION: weather_briefing] [ACTION: {name}]")
        self.assertEqual(self.spoken[0], lead)

    def test_fire_and_exit_with_a_spoken_result_is_still_spoken(self):
        # Even if a fire-and-exit action ever returned a speakable verbatim
        # line, its lead-in may be the last thing the owner hears before the
        # process exits: never skipped.
        bc = self.bc
        name = sorted(bc._FIRE_AND_EXIT_ACTIONS)[0]
        self._p(bc, "SPEAK_RESULT_VERBATIM_ACTIONS",
                set(bc.SPEAK_RESULT_VERBATIM_ACTIONS) | {name})
        bc.ACTIONS[name] = lambda a="": "Going down now."
        printed = self._run(f"{_LEAD} [ACTION: {name}]")
        self.assertEqual(self.spoken[0], _LEAD)
        self.assertNotIn("[answer-first]", printed)

    def test_long_verbatim_answer_keeps_lead(self):
        # A long answer takes seconds to synthesise; the lead-in is the only
        # quick "I heard you".
        self.assertGreater(len(_LONG_ANSWER.split()), 30)
        self.bc.ACTIONS["weather_briefing"] = lambda a="": _LONG_ANSWER
        self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LEAD, _LONG_ANSWER])

    def test_long_verbatim_answer_after_filler_spoke_drops_lead(self):
        self.bc.ACTIONS["weather_briefing"] = lambda a="": _LONG_ANSWER
        turn = self._arm_filler()
        turn.fired.add(1)          # "Processing, sir." already played
        self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LONG_ANSWER])

    def test_long_verbatim_answer_with_pending_filler_keeps_lead(self):
        # The answer's own _speak turns stage 1 off before it synthesises, so
        # a filler that has not spoken yet does not cover the long synth.
        self.bc.ACTIONS["weather_briefing"] = lambda a="": _LONG_ANSWER
        self._arm_filler()
        self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_LEAD, _LONG_ANSWER])


class InformativeAnswerFirstTests(_Base):
    def setUp(self):
        super().setUp()
        self.followup.side_effect = ["It's a text editor, sir.", ""]

    def test_filler_armed_drops_lead(self):
        self._arm_filler()
        self._run(f"{_LEAD} [ACTION: see_screen]", text="what's on screen")
        self.assertEqual(self.spoken, ["It's a text editor, sir."])
        self.assertIn(_LEAD, self._history_text())

    def test_typed_turn_keeps_lead(self):
        self._run(f"{_LEAD} [ACTION: see_screen]", text="what's on screen")
        self.assertEqual(self.spoken, [_LEAD, "It's a text editor, sir."])

    def test_filler_armed_by_another_thread_keeps_lead(self):
        f = self.bc._processing_filler
        box = []
        th = threading.Thread(target=lambda: box.append(f.arm()))
        th.start()
        th.join()
        self.addCleanup(f.disarm, box[0])
        self.assertTrue(f.armed())
        self._run(f"{_LEAD} [ACTION: see_screen]", text="what's on screen")
        self.assertEqual(self.spoken[0], _LEAD)

    def test_filler_cancelled_by_long_running_action_keeps_lead(self):
        # An informative action that is ALSO long-running cancels the filler
        # (mid-task status owns that silence), so the filler will never say
        # "I heard you" — the lead-in must stay.
        bc = self.bc
        self._p(bc, "MID_TASK_STATUS_ENABLED", True)
        self._p(bc, "_emit_mid_task_status")
        self._p(bc, "INFORMATIVE_ACTIONS",
                set(bc.INFORMATIVE_ACTIONS) | {"lookup_record"})
        self._p(bc, "LONG_RUNNING_ACTIONS",
                set(bc.LONG_RUNNING_ACTIONS) | {"lookup_record"})
        bc.ACTIONS["lookup_record"] = lambda a="": "Record located."
        turn = self._arm_filler()
        printed = self._run(f"{_LEAD} [ACTION: lookup_record, someone]",
                            text="look them up")
        self.assertTrue(turn.cancel.is_set())
        self.assertEqual(self.spoken, [_LEAD, "It's a text editor, sir."])
        self.assertNotIn("[answer-first]", printed)

    def test_cold_clip_cache_keeps_lead(self):
        # Armed, but no first-stage clip is cached yet (boot warm still
        # running, or a voice change pruned it): the filler cannot speak.
        self._p(self.bc, "_filler_clips", self._clips(warm=False))
        self._arm_filler()
        self._run(f"{_LEAD} [ACTION: see_screen]", text="what's on screen")
        self.assertEqual(self.spoken, [_LEAD, "It's a text editor, sir."])

    def test_suppressed_filler_keeps_lead(self):
        self.bc._filler_suppressed.return_value = "tray-mute"
        self._arm_filler()
        self._run(f"{_LEAD} [ACTION: see_screen]", text="what's on screen")
        self.assertEqual(self.spoken[0], _LEAD)

    def test_long_filler_delay_keeps_lead(self):
        # A user-set delay longer than the follow-up round: the answer would
        # arrive before the filler ever spoke.
        self._p(self.bc, "_processing_filler", self._filler(first=10.0))
        self._arm_filler()
        self._run(f"{_LEAD} [ACTION: see_screen]", text="what's on screen")
        self.assertEqual(self.spoken[0], _LEAD)

    def test_voice_dispatch_with_filler_drops_lead(self):
        # Wiring through the real wrapper: a voice turn arms the filler, the
        # body sees it armed on its own thread.
        bc = self.bc
        self._p(bc, "PROCESSING_FILLER_ENABLED", True)
        self._p(bc, "_realtime_session", [None])
        self._p(bc, "_filler_warm_if_needed")
        reply = f"{_LEAD} [ACTION: see_screen]"
        self._p(bc, "get_response_with_animation", return_value=reply)
        with contextlib.redirect_stdout(io.StringIO()):
            bc._run_llm_dispatch("what's on screen", voice=True)
        self.assertEqual(self.spoken, ["It's a text editor, sir."])
        self.assertFalse(bc._processing_filler.armed())

    def test_typed_dispatch_keeps_lead(self):
        bc = self.bc
        self._p(bc, "PROCESSING_FILLER_ENABLED", True)
        self._p(bc, "_filler_warm_if_needed")
        reply = f"{_LEAD} [ACTION: see_screen]"
        self._p(bc, "get_response_with_animation", return_value=reply)
        with contextlib.redirect_stdout(io.StringIO()):
            bc._run_llm_dispatch("what's on screen")
        self.assertEqual(self.spoken, [_LEAD, "It's a text editor, sir."])


class TagAndQuipTests(_Base):
    def test_tag_only_lead_is_not_reported_as_dropped(self):
        printed = self._run("[intent:acknowledge] [ACTION: weather_briefing]")
        self.assertNotIn("[answer-first]", printed)
        self.assertEqual(self.spoken[-1], _ANSWER)

    def test_tagged_fifteen_word_lead_is_dropped(self):
        printed = self._run(f"[intent:acknowledge] [mood:calm_efficient] "
                            f"{_ACK_15} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])
        self.assertIn("[answer-first] dropped lead-in (15 words)", printed)

    def test_wry_tagged_lead_is_dropped(self):
        printed = self._run(f"[wry] {_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])
        self.assertIn("[answer-first] dropped lead-in (3 words)", printed)

    def test_decision_is_made_before_the_quip(self):
        # A 13-word acknowledgement plus a random quip aside is 16 words; the
        # roll must not decide whether the lead-in is spoken.
        bc = self.bc
        self._p(bc, "_apply_quip_layer",
                side_effect=lambda s, r: f"{s} As ever, sir.")
        lead = ("Of course, sir, I will look into that for you right away "
                "now.")
        self.assertEqual(len(lead.split()), 13)
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [_ANSWER])

    def test_kept_lead_still_gets_the_quip(self):
        bc = self.bc
        self._p(bc, "_apply_quip_layer",
                side_effect=lambda s, r: f"{s} As ever, sir.")
        lead = "Paris is the capital of France, sir."
        self._run(f"{lead} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken, [f"{lead} As ever, sir.", _ANSWER])


class UnchangedPathsTests(_Base):
    def test_no_action_is_spoken(self):
        self._run(_LEAD)
        self.assertEqual(self.spoken, [_LEAD])

    def test_pushback_objection_is_spoken(self):
        bc = self.bc
        objection = "I'd rather hold off on that one, sir."
        bc.ACTIONS["close_window"] = lambda a="": "closed"
        bc._jarvis_pushback.side_effect = (
            lambda n, a: (objection, "test") if n == "close_window" else None)
        self.addCleanup(bc._pending_confirmation.clear)
        self._run("[ACTION: weather_briefing] [ACTION: close_window, all]")
        self.assertEqual(self.spoken, [objection, _ANSWER])

    def test_confirmation_prompt_is_spoken(self):
        bc = self.bc
        bc.ACTIONS["wipe_cache"] = lambda a="": "wiped"
        bc._needs_confirmation.side_effect = lambda n, a: n == "wipe_cache"
        self.addCleanup(bc._pending_confirmation.clear)
        self._run("[ACTION: weather_briefing] [ACTION: wipe_cache]")
        self.assertEqual(len(self.spoken), 2, self.spoken)
        self.assertIn("needs your confirmation", self.spoken[0])
        self.assertEqual(self.spoken[1], _ANSWER)

    def test_autocorrect_ambiguity_keeps_lead(self):
        bc = self.bc
        ac = mock.Mock()
        ac.autocorrect_command_choice.return_value = {
            "status": "ambiguous",
            "primary": ("weather_briefing", 0.81),
            "secondary": ("see_screen", 0.80),
        }
        self._p(bc, "_cmd_autocorrect", ac)
        self.addCleanup(bc._pending_autocorrect_choice.clear)
        printed = self._run(f"{_LEAD} [ACTION: weathr_screen] "
                            "[ACTION: weather_briefing]")
        self.assertIn("Did you mean", self.spoken[0])
        self.assertIn(_LEAD, self.spoken)
        self.assertNotIn("[answer-first]", printed)

    def test_preemptive_injection_keeps_prose(self):
        # Token-less prose that the hallucination layer maps to a real action:
        # the prose is spoken exactly as before, then the true answer.
        bc = self.bc
        self._p(bc, "_detect_preemptive_hallucination",
                return_value=("inject", "weather_briefing", "test"))
        lead = "Lovely weather out there, sir."
        self._run(lead)
        self.assertEqual(self.spoken, [lead, _ANSWER])

    def test_dropped_step_warning_keeps_lead(self):
        bc = self.bc
        self._p(bc, "_detect_dropped_steps",
                return_value=[("see_screen", "look at the screen")])
        self._arm_filler()
        self._run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(self.spoken[0], _LEAD)


class TurnTimingFieldTests(_Base):
    def _timed_run(self, reply, during_llm=None):
        bc = self.bc
        from core import turn_timing as tt
        lines = []
        timing = tt.TurnTiming(print_fn=lines.append)
        self._p(bc, "_turn_timing", timing)
        timing.begin("inject")
        printed = self._run(reply, during_llm=during_llm)
        timing.emit("ok")
        self.assertEqual(len(lines), 1)
        return tt.parse_line(lines[0]), printed

    def test_lead_dropped_1(self):
        d, _ = self._timed_run(f"{_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(d["lead_dropped"], "1")

    def test_lead_dropped_0(self):
        d, _ = self._timed_run(f"{_LONG_LEAD} [ACTION: weather_briefing]")
        self.assertEqual(d["lead_dropped"], "0")

    def test_tag_only_lead_is_lead_dropped_0(self):
        d, _ = self._timed_run("[intent:acknowledge] [ACTION: weather_briefing]")
        self.assertEqual(d["lead_dropped"], "0")

    def test_barged_turn_reports_nothing_dropped(self):
        # A wake-word barge accepted while the reply streamed silences the
        # rest of the turn; the answer-first rule is not involved.
        bc = self.bc

        def barge():
            bc._tts_interrupt_seq[0] += 1

        d, printed = self._timed_run(f"{_LEAD} [ACTION: weather_briefing]",
                                     during_llm=barge)
        self.assertEqual(d["lead_dropped"], "0")
        self.assertNotIn("[answer-first]", printed)
        self.assertEqual(self.spoken, [])


class HelperTests(_Base):
    def test_helper_never_raises(self):
        bc = self.bc
        self.assertEqual(
            bc._answer_first_drop_count(None, None, None, "pending"), 0)
        self.assertEqual(
            bc._answer_first_drop_count("[ACTION: x]", "Ok sir.",
                                        [object()], "pending"), 0)

    def test_filler_probe_ignores_mocks(self):
        self._p(self.bc, "_processing_filler", mock.Mock())
        self.assertEqual(self.bc._answer_first_filler_state(), "")

    def test_audible_lead_strips_leading_tags_only(self):
        f = self.bc._answer_first_audible_lead
        self.assertEqual(f("[intent:acknowledge] [wry] [mood:dry_amused] Hi."),
                         "Hi.")
        self.assertEqual(f("[intent:acknowledge]  "), "")
        self.assertEqual(f("Hi [wry] there."), "Hi [wry] there.")

    def test_acknowledgement_lexicon(self):
        ack = self.bc._answer_first_is_ack
        for s in ("One moment, sir.", "On it, sir.", "Right away, sir.",
                  "Certainly, sir.", "Of course.", "Very good, sir.",
                  "Let me check, sir.", "Allow me, sir.",
                  "I’ll check on that for you, sir.",
                  "Pulling that up now, sir.",
                  "Let me check your graphics card for you."):
            self.assertTrue(ack(s), s)
        for s in ("Paris is the capital of France.", "Timer set, sir.",
                  "Of course not, sir.", "Certainly, the capital is Paris.",
                  "Let me check, but I won't delete them.",
                  "Let me check why it won't start.",
                  "Checking now - it is charging.", "Sir.", ""):
            self.assertFalse(ack(s), s)


if __name__ == "__main__":
    unittest.main()
