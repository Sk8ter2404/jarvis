"""Monolith wiring of the post-hoc claim validators (2026-09-29 quality sweep).

Three live v2.0.115 false positives, each of which cost an extra LLM round
and extra speech on a turn where nothing was wrong:

  (A) "tell me something interesting" -> "... the moon is actually moving
      about 1.5 inches away ..." matched the reactive check's bare "moving "
      substring; the forced follow-up round retracted "I can't actually move
      the moon".
  (B) "what did I just ask you" -> "On it, sir. You asked about <x>." matched
      "on it, sir"; the second round gave a second, wrong answer.
  (C) "what's the date tomorrow" -> get_time ran, and the follow-up round's
      "It is 2:53 PM ..." made the preemptive injector run get_time AGAIN.

Each fix is pinned in both directions: the false positive is gone AND a real
hallucination is still caught. The reactive rules themselves are unit-tested
in tests/test_claim_validator.py; this module drives the REAL
parse_and_run_actions / _run_llm_dispatch with every LLM call, action and
speech output stubbed (no network, no audio). Generic fixtures only.

    python -m unittest tests.monolith.test_monolith_claim_validation
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

MOON = ("[intent:amused] If I may say so, sir... the moon is actually moving "
        "about 1.5 inches away from us every year.")
ACK_ANSWER = ("[intent:confirmation] On it, sir. You asked about the desk "
              "speaker's battery.")
TIME_RESULT = "current time is 02:53 PM on Tuesday, September 29, 2026"
TIME_READBACK = ("[intent:confirmation] It is 2:53 PM on Tuesday, "
                 "September 29, 2026, sir.")


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoken: list[str] = []
        self._p(bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        self._p(bc, "_cmd_autocorrect", None)
        self._p(bc, "_draft_preview_gate", None)
        self._p(bc, "_processing_filler", mock.Mock())
        self._p(bc, "PC_CONTROL_ENABLED", True)
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        self._p(bc, "MID_TASK_STATUS_ENABLED", False)
        self._p(bc, "_needs_confirmation", lambda n, a: False)
        self._p(bc, "_jarvis_pushback", lambda n, a: None)
        self._p(bc, "maybe_glance_response", return_value=None)
        self._p(bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        self._p(bc, "_tts_interrupt_seq", [0])
        self._p(bc, "_stream_spoken_prefix", [""])
        self.calls: dict[str, list[str]] = {}
        self._actions = dict(bc.ACTIONS)
        self._p(bc, "ACTIONS", self._actions)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _quiet(self, fn, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    def _stub(self, name, *results):
        """Register a stub action returning ``results`` in turn (the last
        one repeats) and recording every call."""
        seq = list(results) or ["ok"]
        self.calls[name] = []

        def fn(arg=""):
            self.calls[name].append(arg)
            i = min(len(self.calls[name]), len(seq)) - 1
            return seq[i]

        self._actions[name] = fn
        return fn

    def _dispatch(self, user_text, first_reply, followups=()):
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=first_reply)
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=list(followups) + [None] * 8)
        return self._quiet(bc._run_llm_dispatch, user_text)

    def _followup_names(self, call_index=0):
        return [n for n, _r in self.gfr.call_args_list[call_index].args[0]]


# ════════════════════════════════════════════════════════════════════════════
#  (A) a third-party fact is not a claim — no retraction round
# ════════════════════════════════════════════════════════════════════════════
class ThirdPartyFactTests(_Base):
    def test_moon_fact_gets_no_validation_result(self):
        cleaned, results = self._quiet(self.bc.parse_and_run_actions, MOON)
        self.assertEqual(results, [])
        self.assertEqual(cleaned, MOON)

    def test_moon_fact_turn_is_one_round_and_one_answer(self):
        self._dispatch("tell me something interesting", MOON,
                       ["[intent:dry_wit] I can't actually move the moon."])
        self.gfr.assert_not_called()
        self.assertEqual(self.spoken, [MOON])

    def test_first_person_claims_still_get_the_correction_round(self):
        for reply in ("Restarting now, sir.", "Moving it now, sir.",
                      "I've moved the window to your left monitor, sir.",
                      "I'll move it for you.", "I have sent the email, sir.",
                      "I've turned off the lights."):
            with self.subTest(reply=reply):
                cleaned, results = self._quiet(self.bc.parse_and_run_actions,
                                               reply)
                self.assertEqual([n for n, _m, _i in results],
                                 ["_unverified_claim"])
                self.assertIn("hallucinated execution", results[0][1])
                self.assertTrue(results[0][2])

    def test_claim_turn_still_runs_the_correction_round(self):
        self._dispatch("restart yourself", "Restarting now, sir.",
                       ["I'm afraid I can't do that from here, sir."])
        self.gfr.assert_called_once()
        self.assertEqual(self._followup_names(), ["_unverified_claim"])


# ════════════════════════════════════════════════════════════════════════════
#  (B) an acknowledgement preface on an answer — one answer, no second round
# ════════════════════════════════════════════════════════════════════════════
class AckPrefaceTests(_Base):
    def test_live_ack_answer_is_accepted_and_the_preface_dropped(self):
        self._dispatch("what did I just ask you", ACK_ANSWER,
                       ["It was about the battery, sir."])
        self.gfr.assert_not_called()
        self.assertEqual(
            self.spoken,
            ["[intent:confirmation] You asked about the desk speaker's "
             "battery."])

    def test_ack_answer_to_a_command_still_runs_the_correction_round(self):
        self._dispatch("turn off the lights",
                       "On it, sir. The lights are off.",
                       ["I'm afraid I can't reach the lights, sir."])
        self.gfr.assert_called_once()
        self.assertEqual(self._followup_names(), ["_unverified_claim"])
        # Not a question, so the preface is spoken untouched.
        self.assertEqual(self.spoken[0], "On it, sir. The lights are off.")

    def test_bare_ack_on_a_question_is_still_a_claim(self):
        # ("Done, sir." is NEW coverage: the old phrase list never had it.)
        for reply in ("On it, sir.", "Right away, sir.", "Done, sir."):
            with self.subTest(reply=reply):
                self.spoken.clear()
                self._dispatch("what's on my screen", reply,
                               ["I can't see it from here, sir."])
                self.gfr.assert_called_once()
                self.assertEqual(self._followup_names(), ["_unverified_claim"])
                self.assertEqual(self.spoken[0], reply)

    def test_action_narration_on_a_question_is_still_a_claim(self):
        for reply in ("Opening Spotify now, sir.",
                      "On it, sir. Taking a look now."):
            with self.subTest(reply=reply):
                self._dispatch("what's playing", reply,
                               ["I can't check that, sir."])
                self.gfr.assert_called_once()
                self.assertEqual(self._followup_names(), ["_unverified_claim"])

    def test_ack_outside_a_dispatch_is_judged_as_before(self):
        # No turn is active (direct call / proactive path): the owner's
        # utterance is unknown, so an ack preface still counts as a claim.
        _c, results = self._quiet(self.bc.parse_and_run_actions, ACK_ANSWER)
        self.assertEqual([n for n, _m, _i in results], ["_unverified_claim"])


# ════════════════════════════════════════════════════════════════════════════
#  (C) a follow-up reading back this turn's result is grounded
# ════════════════════════════════════════════════════════════════════════════
class PreemptiveGroundingTests(_Base):
    def test_followup_time_readback_does_not_rerun_get_time(self):
        self._stub("get_time", TIME_RESULT)
        self._dispatch("what's the date tomorrow", "[ACTION: get_time]",
                       [TIME_READBACK])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.gfr.assert_called_once()
        self.assertIn(TIME_READBACK, self.spoken)

    def test_changed_result_no_longer_buys_an_extra_llm_round(self):
        # When the clock ticked between the two get_time runs the re-injected
        # result was NEW information, so the old code paid a second follow-up
        # LLM round (and a second spoken time) for it.
        self._stub("get_time", TIME_RESULT, TIME_RESULT.replace("53", "54"))
        self._dispatch("what's the date tomorrow", "[ACTION: get_time]",
                       [TIME_READBACK, "It is 2:54 PM, sir."])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.gfr.assert_called_once()

    def test_first_round_time_from_memory_is_still_injected(self):
        self._stub("get_time", TIME_RESULT)
        self._dispatch("what time is it", "It's 1:47 AM, sir.",
                       ["Rather late, sir."])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.gfr.assert_called_once()
        self.assertEqual(self._followup_names(), ["get_time"])

    def test_failed_get_time_grounds_nothing(self):
        # get_time FAILED in round 1, so a time stated in the follow-up is not
        # a read-back: the real action is injected again.
        self._stub("get_time", "could not read the clock", TIME_RESULT)
        self._dispatch("what time is it", "[ACTION: get_time]",
                       ["It's 1:47 AM, sir.", "Rather late, sir."])
        self.assertEqual(len(self.calls["get_time"]), 2)

    def test_other_claim_category_is_still_injected(self):
        # get_time grounds a clock time, not the weather.
        self._stub("get_time", TIME_RESULT)
        self._stub("weather_briefing", "Currently 64 degrees and clear, sir.")
        self._dispatch("what time is it", "[ACTION: get_time]",
                       ["It's 72 degrees and sunny outside, sir."])
        self.assertEqual(len(self.calls["weather_briefing"]), 1)

    def test_grounded_time_does_not_hide_an_invented_weather_claim(self):
        # The time is read back (grounded) but the weather is made up: the
        # scan skips the grounded category and injects weather_briefing.
        # (The old code stopped at the first match and re-ran get_time.)
        self._stub("get_time", TIME_RESULT)
        self._stub("weather_briefing", "Currently 64 degrees and clear, sir.")
        self._dispatch("what time is it", "[ACTION: get_time]",
                       ["It's 2:53 PM and 72 degrees and sunny outside, sir."])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.assertEqual(len(self.calls["weather_briefing"]), 1)

    def test_alias_with_the_same_handler_grounds_the_claim(self):
        fn = self._stub("get_time", TIME_RESULT)
        self._actions["clock_alias"] = fn
        self._dispatch("what time is it", "[ACTION: clock_alias]",
                       [TIME_READBACK])
        self.assertEqual(len(self.calls["get_time"]), 1)

    def test_grounding_never_outlives_the_turn(self):
        bc = self.bc
        self._stub("get_time", TIME_RESULT)
        self._dispatch("what time is it", "[ACTION: get_time]", [TIME_READBACK])
        self.assertEqual(bc._turn_actions_ran(), frozenset())
        self.assertEqual(bc._turn_user_text(), "")
        # A later reply outside any turn is judged from scratch again.
        _c, results = self._quiet(bc.parse_and_run_actions,
                                  "It's 1:47 AM, sir.")
        self.assertEqual([n for n, _r, _i in results], ["get_time"])
        self.assertEqual(len(self.calls["get_time"]), 2)


class PreemptiveDetectorPredicateTests(_Base):
    def test_grounded_inject_is_skipped_and_the_scan_continues(self):
        bc = self.bc
        self._stub("weather_briefing", "stub")
        reply ="It's 2:53 PM and 72 degrees and sunny outside, sir."
        self.assertEqual(bc._detect_preemptive_hallucination(reply)[1],
                         "get_time")
        out = bc._detect_preemptive_hallucination(
            reply, is_grounded=lambda a: a == "get_time")
        self.assertEqual(out[:2], ("inject", "weather_briefing"))
        self.assertIsNone(bc._detect_preemptive_hallucination(
            "It's 2:53 PM, sir.", is_grounded=lambda a: True))

    def test_refuse_patterns_are_never_grounded(self):
        out = self.bc._detect_preemptive_hallucination(
            "Panning the camera to the left now, sir.",
            is_grounded=lambda a: True)
        self.assertEqual(out[0], "refuse")


class ReactiveGroundingTests(_Base):
    def test_followup_summary_of_the_action_is_not_a_claim(self):
        self._stub("play_music", "playing Blue Horizon by the Example Quartet")
        self._dispatch("play some jazz", "[ACTION: play_music, jazz]",
                       ["Playing Blue Horizon by the Example Quartet, sir.",
                        "I can't play that, sir."])
        self.gfr.assert_called_once()
        self.assertEqual(len(self.calls["play_music"]), 1)

    def test_followup_claiming_a_different_action_is_still_caught(self):
        self._stub("get_time", TIME_RESULT)
        self._dispatch("what time is it", "[ACTION: get_time]",
                       ["Opening your calendar now, sir.",
                        "I can't open it, sir."])
        self.assertEqual(self.gfr.call_count, 2)
        self.assertEqual(self._followup_names(1), ["_unverified_claim"])

    def test_grounded_time_readback_does_not_hide_a_new_claim(self):
        # The old code injected get_time for the read-back time, which gave
        # the reply a result and so skipped the claim check: the invented
        # "opening your calendar" went unchallenged. Now the time is grounded
        # and the claim is caught.
        self._stub("get_time", TIME_RESULT)
        self._dispatch("what time is it", "[ACTION: get_time]",
                       ["It is 2:53 PM. Opening your calendar now, sir.",
                        "I can't open it, sir."])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.assertEqual(self.gfr.call_count, 2)
        self.assertEqual(self._followup_names(1), ["_unverified_claim"])


# ════════════════════════════════════════════════════════════════════════════
#  the ledger itself
# ════════════════════════════════════════════════════════════════════════════
class TurnGroundingLedgerTests(_Base):
    def test_records_successes_only_and_only_inside_a_turn(self):
        bc = self.bc
        bc._note_turn_action_ran("get_time", TIME_RESULT)   # no turn: ignored
        self.assertEqual(bc._turn_actions_ran(), frozenset())
        prev = bc._begin_turn_grounding("what time is it")
        try:
            bc._note_turn_action_ran("get_time", TIME_RESULT)
            bc._note_turn_action_ran("see_screen", "could not capture screen")
            self.assertEqual(bc._turn_actions_ran(), frozenset({"get_time"}))
            self.assertEqual(bc._turn_user_text(), "what time is it")
        finally:
            bc._end_turn_grounding(prev)
        self.assertEqual(bc._turn_actions_ran(), frozenset())

    def test_nested_turn_restores_the_outer_ledger(self):
        bc = self.bc
        outer = bc._begin_turn_grounding("outer")
        try:
            bc._note_turn_action_ran("get_time", TIME_RESULT)
            inner = bc._begin_turn_grounding("inner")
            self.assertEqual(bc._turn_actions_ran(), frozenset())
            bc._end_turn_grounding(inner)
            self.assertEqual(bc._turn_actions_ran(), frozenset({"get_time"}))
            self.assertEqual(bc._turn_user_text(), "outer")
        finally:
            bc._end_turn_grounding(outer)

    def test_ledger_closes_even_when_the_turn_raises(self):
        bc = self.bc
        self._p(bc, "_run_llm_dispatch_body", side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            self._quiet(bc._run_llm_dispatch, "what time is it")
        self.assertEqual(bc._turn_user_text(), "")


# ════════════════════════════════════════════════════════════════════════════
#  (D) v2.0.148: a clock time stated for ANOTHER place is checked
# ════════════════════════════════════════════════════════════════════════════
# Live v2.0.140 (Tue 22:17 US Central): "what time is it in London" ran
# get_time (the LOCAL clock) and the follow-up said "It is 10:17 PM in London,
# sir." — grounded by get_time as far as the preemptive layer knew. London
# was at 4:17 AM. The fast path answers the plain question first; this guard
# covers everything that still reaches the LLM.
LOCAL_TIME_RESULT = "current time is 10:17 PM on Tuesday, September 29, 2026"
LIVE_LONDON = "It is 10:17 PM in London, sir."
TRUE_LONDON = "It's 4:17 AM in London, sir. That's Wednesday there."


class _InlineThread:
    def __init__(self, target=None, args=(), kwargs=None, **_kw):
        self._t, self._a, self._k = target, args, kwargs or {}

    def start(self):
        if self._t:
            self._t(*self._a, **self._k)

    def join(self, *_a, **_k):
        return None

    def is_alive(self):
        return False


class WorldClockGuardTests(_Base):
    def setUp(self):
        super().setUp()
        try:
            import datetime as dt
            from zoneinfo import ZoneInfo
            now = dt.datetime(2026, 9, 29, 22, 17,
                              tzinfo=ZoneInfo("America/Chicago"))
        except Exception:   # pragma: no cover - no zone data
            self.skipTest("no IANA time zone data")
        self._p(self.bc, "_fast_path_now", return_value=now)

    def test_live_followup_is_corrected_not_read_back(self):
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._dispatch("what time is it in London", "[ACTION: get_time]",
                       [LIVE_LONDON, "Rather early there, sir."])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.gfr.assert_called_once()
        self.assertIn(TRUE_LONDON, self.spoken)
        self.assertFalse(any("10:17 PM in London" in s for s in self.spoken))

    def test_wrong_time_from_memory_is_corrected_without_get_time(self):
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._dispatch("what's it like in London now", LIVE_LONDON,
                       ["Rather early there, sir."])
        self.assertEqual(self.calls["get_time"], [])
        self.gfr.assert_not_called()
        self.assertEqual(self.spoken, [TRUE_LONDON])

    def test_right_time_is_left_alone_and_grounded(self):
        self._stub("get_time", LOCAL_TIME_RESULT)
        reply = "It's 4:17 AM in London, sir."
        cleaned, results = self._quiet(self.bc.parse_and_run_actions, reply)
        self.assertEqual((cleaned, results), (reply, []))
        self.assertEqual(self.calls["get_time"], [])

    def test_other_actions_in_the_reply_still_run(self):
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._stub("set_timer", "timer set for 5 minutes")
        cleaned, results = self._quiet(
            self.bc.parse_and_run_actions,
            "[intent:confirmation] It is 10:17 PM in London, sir. "
            "[ACTION: get_time] [ACTION: set_timer, 5 minutes]")
        self.assertEqual(self.calls["get_time"], [])
        self.assertEqual(self.calls["set_timer"], ["5 minutes"])
        self.assertIn("It's 4:17 AM in London", cleaned)
        self.assertNotIn("10:17 PM", cleaned)

    def test_a_local_time_claim_is_judged_as_before(self):
        # No place: the preemptive layer still injects get_time.
        self._stub("get_time", LOCAL_TIME_RESULT)
        _c, results = self._quiet(self.bc.parse_and_run_actions,
                                  "It's 1:47 AM, sir.")
        self.assertEqual([n for n, _r, _i in results], ["get_time"])

    def test_the_early_speech_flush_holds_a_place_time_claim(self):
        # "The time in London is ..." matches no preemptive pattern, so
        # without the world-clock hold it would be voiced before the guard.
        bc = self.bc
        for first in ("The time in London is 10:17 PM, sir. ",
                      "In London, it's 10:17 PM, sir. "):
            with self.subTest(first=first):
                spoken: list[str] = []
                with mock.patch.object(bc, "threading", mock.Mock(
                        Thread=_InlineThread)):
                    buf = bc._SentenceFlushBuffer(speak_fn=spoken.append)
                    for c in (first, "Rather early there. ", "tail"):
                        buf.feed(c)
                self.assertEqual(spoken, [])

    def test_a_guard_failure_changes_nothing(self):
        self._p(self.bc._world_clock, "check_time_claim",
                side_effect=RuntimeError("boom"))
        self._stub("get_time", LOCAL_TIME_RESULT)
        _c, results = self._quiet(self.bc.parse_and_run_actions,
                                  "It's 1:47 AM, sir.")
        self.assertEqual([n for n, _r, _i in results], ["get_time"])

    # ── Review WC-1: a conversion / plan / event time is not "now" ────────

    def test_a_conversion_answer_is_spoken_not_replaced(self):
        # "if it's 9 AM here what time is it in London" reaches the LLM (the
        # fast path does not answer it); its right answer used to be thrown
        # away for the current London time.
        reply = "When it's 9 AM here, it's 3 PM in London, sir."
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._dispatch("if it's 9 AM here what time is it in London", reply)
        self.assertIn(reply, self.spoken)
        self.assertFalse(any("4:17 AM in London" in s for s in self.spoken))

    def test_a_reminder_confirmation_is_kept(self):
        self._stub("set_reminder", "reminder set for 3 AM")
        cleaned, results = self._quiet(
            self.bc.parse_and_run_actions,
            "[ACTION: set_reminder, 3 AM] I'll remind you when it's 9 AM in "
            "London, sir.")
        self.assertEqual(self.calls["set_reminder"], ["3 AM"])
        self.assertIn("I'll remind you when it's 9 AM in London", cleaned)
        self.assertNotIn("4:17 AM", cleaned)

    def test_event_and_future_times_are_never_rewritten(self):
        for reply in (
                "If you call at 8 PM, it'll be 2 AM in London, sir.",
                "When you land in London, it'll be 6 AM, sir.",
                "The local time in Tokyo will be 3 PM when you land, sir.",
                "It is 3 PM in London on Saturday when the match kicks off, "
                "sir.",
                "Your call with the Boston office? It's 3 PM Eastern time, "
                "sir.",
                "It's 9 AM Pacific time when the stream starts, sir."):
            with self.subTest(reply=reply):
                cleaned, _results = self._quiet(
                    self.bc.parse_and_run_actions, reply)
                self.assertTrue(cleaned.startswith(reply.split(",")[0]),
                                cleaned)
                self.assertNotIn("That's Wednesday there", cleaned)

    # ── Review WC-2 / WC-3 / WC-4: a correct answer is never replaced ────

    def test_correct_answers_for_unmapped_places_are_kept(self):
        for reply in ("It's 6:17 AM in Eastern Europe, sir.",
                      "It's 10:17 PM in Athens, Georgia, sir.",
                      "It's 11:17 PM in London, Ontario, sir.",
                      "It's 10:17 PM. In London, it's 4:17 AM."):
            with self.subTest(reply=reply):
                cleaned, _results = self._quiet(
                    self.bc.parse_and_run_actions, reply)
                self.assertEqual(cleaned, reply)

    # ── Review WC-5: a right place-time never vouches for a wrong local one

    def test_a_wrong_local_time_beside_a_right_london_time_gets_get_time(self):
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._dispatch("what time is it here and in London",
                       "It's 10:02 PM here, sir, and in London it's about "
                       "4 AM.",
                       ["It's 10:17 PM here, sir, and 4:17 AM in London."])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.gfr.assert_called_once()

    # ── Second review: residual WC-1 / WC-3 / WC-4, end to end ────────────

    def test_a_time_difference_answer_is_spoken_not_replaced(self):
        # A sentence with a second clock time that is not the time here now
        # is a conversion; "time difference" questions reach the LLM.
        reply = ("Tokyo is 14 hours ahead of you, sir, so at 9 AM here it's "
                 "11 PM in Tokyo.")
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._dispatch("what is the time difference between here and Tokyo",
                       reply)
        self.assertIn(reply, self.spoken)
        self.assertFalse(any("12:17 PM in Tokyo" in s for s in self.spoken))

    def test_an_abbreviated_us_state_answer_is_kept(self):
        reply = "It's 11:17 PM in Athens, GA, sir."
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._dispatch("what time is it in Athens Georgia", reply)
        self.assertIn(reply, self.spoken)
        self.assertFalse(any("6:17 AM in Athens" in s for s in self.spoken))

    def test_a_time_now_question_with_a_reason_is_still_corrected(self):
        # The first fix's question blacklist ("call", "game", ...) switched
        # the guard off here, and the live bug was voiced unchanged.
        self._stub("get_time", LOCAL_TIME_RESULT)
        self._dispatch("what time is it in London? I want to call my mom",
                       "[ACTION: get_time]", [LIVE_LONDON])
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.assertIn(TRUE_LONDON, self.spoken)
        self.assertFalse(any("10:17 PM in London" in s for s in self.spoken))

    def test_a_two_clause_answer_with_a_connective_is_kept(self):
        reply = "It's 10:17 PM here, whereas in London, it's 4:17 AM."
        cleaned, _results = self._quiet(self.bc.parse_and_run_actions, reply)
        self.assertEqual(cleaned, reply)

    def test_the_masked_scan_still_injects_for_the_local_clause(self):
        self._stub("get_time", LOCAL_TIME_RESULT)
        _c, results = self._quiet(
            self.bc.parse_and_run_actions,
            "It's 10:02 PM here, sir, and in London it's about 4 AM.")
        self.assertEqual([n for n, _r, _i in results], ["get_time"])


if __name__ == "__main__":
    unittest.main()
