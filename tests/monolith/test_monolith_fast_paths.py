"""Monolith wiring for the deterministic fast paths (core/fast_paths.py,
core/date_math.py; 2026-09-29).

The fast path answers relative-date questions, "what did I just ask you" and
"what's my name" (plus, since v2.0.148, "what's the date next Monday", "what
time is it in London", "who am I" and "what was the first thing I asked you")
right before the LLM dispatch, inside _run_voice_shortcuts,
which main() runs for voice AND typed / injected turns. These tests show it
answers without calling the LLM, records the turn in conversation_history like
a normal turn, never arms the processing filler, logs one "[fast-path] <kind>"
line, and stands down when FAST_PATHS_ENABLED is False. The clock is frozen
through _fast_path_now (CI runs in UTC). Generic fixtures only.

    python -m unittest tests.monolith.test_monolith_fast_paths
"""
from __future__ import annotations

import contextlib
import datetime as dt
import inspect
import io
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

TUE = dt.datetime(2026, 9, 29, 14, 56)


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._speak = self._p(bc, "_speak")
        self._p(bc, "set_state")
        self._p(bc, "_fast_path_now", return_value=TUE)
        self._p(bc, "FAST_PATHS_ENABLED", True)
        self._p(bc, "USER_NAME", "Alex")
        # The LLM must never be reached on a fast-path turn.
        self.llm = self._p(bc, "_call_llm",
                           side_effect=AssertionError("LLM called"))
        self.get_resp = self._p(bc, "get_response_with_animation",
                                side_effect=AssertionError("LLM called"))
        self.arm = self._p(bc._processing_filler, "arm")
        # Every earlier shortcut stands aside (as in test_monolith_sec7).
        self._p(bc, "maybe_replay_last_action", return_value=None)
        router = types.ModuleType("core.mode_router")
        router.maybe_handle_mode_toggle = lambda _t: None
        router.controlled_dispatch = lambda _t, _a: None
        router.is_in_controlled_mode = lambda: False
        disp = types.ModuleType("core.dispatcher")
        disp.resolve_and_dispatch = lambda _t, _a: None
        voice = types.SimpleNamespace(maybe_switch_backend=lambda _t: None)
        patcher = mock.patch.dict(bc.sys.modules, {
            "core.mode_router": router, "core.dispatcher": disp,
            "skill_custom_voice": voice})
        patcher.start()
        self.addCleanup(patcher.stop)
        bc.conversation_history.clear()
        # _session_opening_turns starts empty: the harness deep-restores it.

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _turn(self, text):
        """What main() does with an accepted utterance: the shortcuts first,
        the LLM dispatch only when none handled it."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            handled = self.bc._run_voice_shortcuts(text)
        return handled, buf.getvalue()


class FastPathAnswersTests(_Base):
    def test_date_tomorrow_is_answered_without_the_llm(self):
        handled, log = self._turn("what's the date tomorrow")
        self.assertTrue(handled)
        reply = "Tomorrow is Wednesday, September 30, 2026, sir."
        self._speak.assert_called_once_with(reply)
        self.llm.assert_not_called()
        self.get_resp.assert_not_called()
        self.assertIn("[fast-path] date", log)
        # One line, the kind only.
        self.assertEqual(log.count("[fast-path]"), 1)

    def test_the_reply_is_in_the_transcript_log(self):
        # Live 2026-09-29: fast-path replies were spoken but never printed as
        # a "JARVIS:" line, so the session log (and the run-jarvis driver)
        # showed no answer at all for them.
        _handled, log = self._turn("what's the date tomorrow")
        self.assertIn("JARVIS: Tomorrow is Wednesday, September 30, 2026, sir.", log)

    def test_days_until_christmas_and_friday(self):
        for text, reply in (
                ("how many days until Christmas",
                 "Christmas is 87 days away, on Friday, December 25, sir."),
                ("how long until Friday",
                 "Friday is 3 days away, on October 2, sir.")):
            with self.subTest(text=text):
                self._speak.reset_mock()
                handled, log = self._turn(text)
                self.assertTrue(handled)
                self._speak.assert_called_once_with(reply)
                self.assertIn("[fast-path] days-until", log)
        self.llm.assert_not_called()

    def test_turn_is_recorded_so_follow_ups_work(self):
        self.bc.conversation_history.extend([
            {"role": "user", "content": "good afternoon"},
            {"role": "assistant", "content": "Good afternoon, sir."}])
        self._turn("what's the date tomorrow")
        self.assertEqual(self.bc.conversation_history[-2:], [
            {"role": "user", "content": "what's the date tomorrow"},
            {"role": "assistant",
             "content": "Tomorrow is Wednesday, September 30, 2026, sir."}])

    def test_the_processing_filler_is_never_armed(self):
        # Handled before _run_llm_dispatch, the only place that arms it
        # ("Processing, sir." on a voice turn).
        handled, _ = self._turn("how many days until Christmas")
        self.assertTrue(handled)
        self.arm.assert_not_called()
        self._speak.assert_called_once()

    def test_last_utterance_excludes_the_current_one(self):
        self.bc.conversation_history.extend([
            {"role": "user", "content": "Jarvis, open the project notes."},
            {"role": "assistant", "content": "Opening them now, sir."}])
        handled, log = self._turn("what did I just ask you")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            'You asked me: "open the project notes", sir.')
        self.assertIn("[fast-path] last-utterance", log)
        # Asked again: still the real question, never the recall question.
        self._speak.reset_mock()
        self._turn("what did I just ask you")
        self._speak.assert_called_once_with(
            'You asked me: "open the project notes", sir.')
        self.llm.assert_not_called()

    def test_last_utterance_with_nothing_earlier_says_so(self):
        handled, _ = self._turn("what was my last question")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            "There's no earlier question from you in this conversation, "
            "sir.")

    def test_owner_name_comes_from_user_name(self):
        handled, log = self._turn("what's my name")
        self.assertTrue(handled)
        self._speak.assert_called_once_with("Your name is Alex, sir.")
        self.assertIn("[fast-path] owner-name", log)
        self.llm.assert_not_called()

    # ── v2.0.148: four more live v2.0.140 wrong answers ─────────────────

    def test_who_am_i_comes_from_user_name(self):
        # Live: recognize_face -> "I don't see a face right now, sir."
        for text, reply in (("who am I", "You're Alex, sir."),
                            ("do you know who I am",
                             "Of course, sir. You're Alex.")):
            with self.subTest(text=text):
                self._speak.reset_mock()
                handled, log = self._turn(text)
                self.assertTrue(handled)
                self._speak.assert_called_once_with(reply)
                self.assertIn("[fast-path] owner-identity", log)
        self.llm.assert_not_called()

    def test_next_monday_is_answered_without_the_llm(self):
        # Live: "It will be September 29th, sir." (today).
        handled, log = self._turn("what's the date next Monday")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            "Next Monday is October 5, 2026, sir.")
        self.assertIn("[fast-path] date-of", log)
        self.llm.assert_not_called()

    def test_world_clock_is_answered_without_the_llm(self):
        # Live: get_time (LOCAL) -> "It is 10:17 PM in London, sir."
        try:
            from zoneinfo import ZoneInfo
            now = dt.datetime(2026, 9, 29, 22, 17,
                              tzinfo=ZoneInfo("America/Chicago"))
        except Exception:   # pragma: no cover - no zone data
            self.skipTest("no IANA time zone data")
        self._p(self.bc, "_fast_path_now", return_value=now)
        handled, log = self._turn("what time is it in London")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            "It's 4:17 AM in London, sir. That's Wednesday there.")
        self.assertIn("[fast-path] world-clock", log)
        self.llm.assert_not_called()

    def test_first_thing_asked_comes_from_the_session_record(self):
        # Live: session_memory_recall said it had no access. The main loop
        # records each accepted owner turn (_note_session_owner_utterance)
        # BEFORE the shortcuts run, exactly as main() does.
        bc = self.bc
        # A blue-green handoff tail from the PREVIOUS process sits in the
        # history; it must never be "the first thing".
        bc.conversation_history.extend([
            {"role": "user", "content": "an older process's question"},
            {"role": "assistant", "content": "An older answer, sir."}])
        for text in ("Jarvis", "what's 12 times 12", "open the project notes",
                     "what was the first thing I asked you in this "
                     "conversation"):
            bc._note_session_owner_utterance(text)
        handled, log = self._turn(
            "what was the first thing I asked you in this conversation")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            'The first thing you asked me this session was: "what\'s 12 '
            'times 12", sir.')
        self.assertIn("[fast-path] first-utterance", log)
        self.llm.assert_not_called()

    def test_first_thing_survives_the_history_trim(self):
        bc = self.bc
        bc._note_session_owner_utterance("what's 12 times 12")
        for i in range(bc.MAX_CONVERSATION_HISTORY):
            bc._append_turn(f"question number {i}", f"answer {i}, sir.")
        self.assertNotIn("what's 12 times 12",
                         [m["content"] for m in bc.conversation_history])
        self._turn("what did I ask you first")
        self._speak.assert_called_with(
            'The first thing you asked me this session was: "what\'s 12 '
            'times 12", sir.')


class FastPathFallThroughTests(_Base):
    def test_disabled_flag_falls_through_to_the_llm(self):
        self._p(self.bc, "FAST_PATHS_ENABLED", False)
        for text in ("what's the date tomorrow", "what's my name",
                     "what did I just ask you"):
            with self.subTest(text=text):
                handled, log = self._turn(text)
                self.assertFalse(handled)
                self.assertNotIn("[fast-path]", log)
        self._speak.assert_not_called()
        self.assertEqual(self.bc.conversation_history, [])

    def test_no_configured_name_falls_through(self):
        self._p(self.bc, "USER_NAME", "")
        for text in ("what's my name", "who am I"):
            with self.subTest(text=text):
                handled, _ = self._turn(text)
                self.assertFalse(handled)
        self._speak.assert_not_called()

    def test_presence_questions_stay_with_the_camera_route(self):
        # "who am I" left this list in v2.0.148 (answered from USER_NAME, see
        # test_who_am_i_comes_from_user_name); presence and recognition
        # questions are still a live camera look.
        for text in ("who is this", "who am I looking at",
                     "do you recognize me", "can you see me", "who's here"):
            with self.subTest(text=text):
                handled, _ = self._turn(text)
                self.assertFalse(handled)
        self._speak.assert_not_called()

    def test_commands_fall_through(self):
        for text in ("remind me tomorrow to call the office",
                     "what's the weather tomorrow", "play music until friday",
                     "pause until tomorrow", "wake me up tomorrow"):
            with self.subTest(text=text):
                handled, _ = self._turn(text)
                self.assertFalse(handled)
        self._speak.assert_not_called()

    def test_a_matcher_error_falls_through(self):
        self._p(self.bc._fast_paths, "match",
                side_effect=RuntimeError("boom"))
        handled, log = self._turn("what's the date tomorrow")
        self.assertFalse(handled)
        self.assertIn("[fast-path] failed", log)
        self._speak.assert_not_called()

    def test_earlier_shortcuts_keep_precedence(self):
        self._p(self.bc, "maybe_replay_last_action",
                return_value="Done again, sir.")
        handled, log = self._turn("what's the date tomorrow")
        self.assertTrue(handled)
        self._speak.assert_called_once_with("Done again, sir.")
        self.assertNotIn("[fast-path]", log)


class FastPathWiringTests(_Base):
    """main() cannot run in a test (see test_monolith_turn_timing)."""

    def test_fast_path_is_the_last_shortcut_before_the_llm(self):
        src = inspect.getsource(self.bc._run_voice_shortcuts)
        self.assertTrue(src.rstrip().endswith("return _run_fast_paths(text)"))
        main = inspect.getsource(self.bc.main)
        self.assertLess(main.index("if _run_voice_shortcuts(text):"),
                        main.index("reply = _run_llm_dispatch(text"))

    def test_local_guard_keeps_whats_my_name_off_the_camera(self):
        guard = self.bc._LOCAL_NEVER_GUESS_GUARD
        self.assertIn("\"What's my name\" is NOT a camera look", guard)
        self.assertIn("Never guess a name.", guard)

    def test_main_records_each_owner_turn_before_the_shortcuts(self):
        main = inspect.getsource(self.bc.main)
        you = main.index("_note_owner_turn()")
        rec = main.index("_note_session_owner_utterance(text)", you)
        self.assertLess(rec, main.index("if _run_voice_shortcuts(text):"))
        src = inspect.getsource(self.bc._run_fast_paths)
        self.assertIn("session_turns=list(_session_opening_turns)", src)
        self.assertIn("session_start_lost=bool(_session_opening_lost[0])",
                      src)


class SessionOpeningRecordTests(_Base):
    """_note_session_owner_utterance: the record "what was the first thing I
    asked" reads (conversation_history is trimmed and handoff-seeded)."""

    def test_records_until_the_first_real_utterance(self):
        bc = self.bc
        for text in ("Jarvis", "what did I just ask you", "what's 12 times 12",
                     "open the project notes", "what's the weather"):
            bc._note_session_owner_utterance(text)
        self.assertEqual(bc._session_opening_turns,
                         ["Jarvis", "what did I just ask you",
                          "what's 12 times 12"])

    def test_bounded_when_nothing_real_arrives(self):
        bc = self.bc
        for _ in range(3 * bc._SESSION_OPENING_MAX):
            bc._note_session_owner_utterance("Jarvis")
        self.assertEqual(len(bc._session_opening_turns),
                         bc._SESSION_OPENING_MAX)
        bc._note_session_owner_utterance("what's 12 times 12")
        self.assertEqual(bc._session_opening_turns[-1], "what's 12 times 12")
        self.assertEqual(len(bc._session_opening_turns),
                         bc._SESSION_OPENING_MAX)

    def test_ignores_non_text_and_never_raises(self):
        bc = self.bc
        for bad in (None, "", "   ", 42, ["x"]):
            bc._note_session_owner_utterance(bad)
        self.assertEqual(bc._session_opening_turns, [])
        with mock.patch.object(bc._fast_paths, "first_owner_utterance",
                               side_effect=RuntimeError("boom")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bc._note_session_owner_utterance("what's 12 times 12")
        self.assertIn("[fast-path] session record failed", buf.getvalue())

    def test_a_first_request_mentioning_first_thing_is_the_first_thing(self):
        # Review F2: the record skipped a real opening request that merely
        # mentions "first thing" (the loose recall detector), kept recording,
        # and "what was the first thing I asked you" named the SECOND one.
        bc = self.bc
        for text in ("what's the first thing on my calendar today",
                     "turn on the desk lamp",
                     "what was the first thing I asked you"):
            bc._note_session_owner_utterance(text)
        self.assertEqual(bc._session_opening_turns,
                         ["what's the first thing on my calendar today"])
        handled, _log = self._turn("what was the first thing I asked you")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            'The first thing you asked me this session was: "what\'s the '
            'first thing on my calendar today", sir.')

    def test_each_entry_is_stamped(self):
        bc = self.bc
        with mock.patch.object(bc.time, "time", return_value=1000.0):
            bc._note_session_owner_utterance("Jarvis")
            bc._note_session_owner_utterance("what's 12 times 12")
        self.assertEqual(bc._session_opening_ts, [1000.0, 1000.0])

    def test_forget_since_drops_the_last_hour_only(self):
        # Review F5: forget_last_hour reaches the record through this helper.
        bc = self.bc
        bc._session_opening_turns[:] = ["Jarvis", "what's 12 times 12"]
        bc._session_opening_ts[:] = [100.0, 5000.0]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            removed = bc._forget_session_opening_since(4000.0)
        self.assertEqual(removed, 1)
        self.assertEqual(bc._session_opening_turns, ["Jarvis"])
        self.assertEqual(bc._session_opening_ts, [100.0])
        self.assertIn("forgot 1 recorded opening utterance", buf.getvalue())
        # The forgotten request is never recited afterwards. Changed
        # deliberately (second review, F5/F6 residual): this pinned "You
        # haven't asked me anything else this session yet" — untrue (he had;
        # it was forgotten) — and the next turn then became "the first
        # thing". The start is now reported as lost.
        self.assertEqual(bc._session_opening_lost, [True])
        self._turn("what was the first thing I asked you")
        self._speak.assert_called_once_with(
            "I no longer have the start of this session on record, sir.")

    def test_after_a_forget_the_next_turn_is_never_the_first_thing(self):
        # Second-review repro: forget emptied the record, the next ordinary
        # utterance was recorded, and "what was the first thing I asked you"
        # named it while the history still showed the earlier turns.
        for cutoff_from_now in (3600, None):     # forget_last_hour / reset
            with self.subTest(cutoff=cutoff_from_now):
                bc = self.bc
                self._speak.reset_mock()
                bc._session_opening_lost[:] = [False]
                bc.conversation_history[:] = [
                    {"role": "user", "content": "book the dentist"},
                    {"role": "assistant", "content": "Done, sir."},
                    {"role": "user", "content": "forget the last hour"},
                    {"role": "assistant", "content": "Forgotten, sir."}]
                t0 = bc.time.time()
                bc._session_opening_turns[:] = ["book the dentist"]
                bc._session_opening_ts[:] = [t0 - 600]
                with contextlib.redirect_stdout(io.StringIO()):
                    bc._forget_session_opening_since(
                        None if cutoff_from_now is None
                        else t0 - cutoff_from_now)
                for text in ("turn on the desk lamp", "open spotify"):
                    bc._note_session_owner_utterance(text)
                    bc.conversation_history.extend([
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "Done, sir."}])
                # The record no longer refills after the start is lost.
                self.assertEqual(bc._session_opening_turns, [])
                q = "what was the first thing I asked you"
                bc._note_session_owner_utterance(q)
                self._turn(q)
                self._speak.assert_called_once_with(
                    "I no longer have the start of this session on record, "
                    "sir.")

    def test_forgetting_only_wake_phrases_loses_nothing(self):
        bc = self.bc
        bc._session_opening_turns[:] = ["Jarvis"]
        bc._session_opening_ts[:] = [bc.time.time() - 60]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                bc._forget_session_opening_since(bc.time.time() - 3600), 1)
        self.assertEqual(bc._session_opening_lost, [False])
        bc._note_session_owner_utterance("what's 12 times 12")
        self._turn("what did I ask you first")
        self._speak.assert_called_once_with(
            'The first thing you asked me this session was: "what\'s 12 '
            'times 12", sir.')

    def test_forget_none_clears_everything_and_unstamped_counts_as_now(self):
        bc = self.bc
        # Filled without stamps (another path): treated as now, so a forget
        # drops it rather than keeping what it cannot date.
        bc._session_opening_turns[:] = ["what's 12 times 12"]
        bc._session_opening_ts[:] = []
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                bc._forget_session_opening_since(bc.time.time() - 3600), 1)
        self.assertEqual(bc._session_opening_turns, [])
        bc._session_opening_turns[:] = ["a", "b"]
        bc._session_opening_ts[:] = [1.0, 2.0]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bc._forget_session_opening_since(None), 2)
        self.assertEqual((bc._session_opening_turns, bc._session_opening_ts),
                         ([], []))


class SessionOpeningHandoffTests(_Base):
    """Review F6: the record is THIS process's, but a blue-green handoff is
    meant to be seamless (the conversation tail is carried). Right after a
    swap, "what was the first thing I asked you today" said nothing was
    asked while "what did I just ask you" recalled the pre-swap turn. The
    record now crosses the swap; a payload without it (an older version)
    never claims nothing was asked."""

    Q = "what was the first thing I asked you today"
    TAIL = [{"role": "user", "content": "set a timer for ten minutes"},
            {"role": "assistant", "content": "Timer set, sir."}]

    def _consume(self, payload):
        bc = self.bc
        bgm = mock.MagicMock()
        bgm.RESUME_HANDOFF_FLAG = "--resume-handoff"
        bgm.consume_handoff_state.return_value = payload
        self._p(bc, "_bgm", bgm)
        self._p(bc, "BLUE_GREEN_ROLE", "prod")
        self._p(bc.sys, "argv", ["bobert", "--resume-handoff"])
        with contextlib.redirect_stdout(io.StringIO()):
            bc._consume_blue_green_handoff()

    def test_the_handoff_writes_the_record(self):
        bc = self.bc
        bc._session_opening_turns[:] = ["Jarvis", "what's 12 times 12"]
        bc._session_opening_ts[:] = [10.0, 20.0]
        payload = self._tick_payload()
        self.assertEqual(payload["session_opening_turns"],
                         ["Jarvis", "what's 12 times 12"])
        self.assertEqual(payload["session_opening_ts"], [10.0, 20.0])
        self.assertIs(payload["session_opening_lost"], False)
        # A lost start is carried too (second review, F5/F6 residual).
        bc._session_opening_lost[:] = [True]
        self.assertIs(self._tick_payload()["session_opening_lost"], True)

    def _tick_payload(self):
        """One prod loop tick that sees a handoff signal; the payload it
        writes."""
        bc = self.bc
        bgm = mock.MagicMock()
        bgm.read_version.return_value = "9.9.9"
        bgm.consume_handoff_signal.return_value = {"target_version": "9.9.10"}
        bgm.consume_upgrade_aborted_signal.return_value = None
        bgm.consume_handoff_failure_signal.return_value = None
        with mock.patch.object(bc, "_bgm", bgm), \
                mock.patch.object(bc, "BLUE_GREEN_ROLE", "prod"), \
                mock.patch.object(bc, "_bg_last_heartbeat", [0.0]), \
                mock.patch.object(bc, "_bg_handoff_seen_at", [0.0]), \
                mock.patch.object(bc, "_bg_handoff_grace_s", [10.0]), \
                contextlib.redirect_stdout(io.StringIO()):
            bc._blue_green_loop_tick()
        return bgm.write_handoff_state.call_args[0][0]

    def test_the_next_process_answers_from_the_carried_record(self):
        bc = self.bc
        self._consume({"conversation_tail": list(self.TAIL),
                       "session_opening_turns": ["Jarvis",
                                                 "what's 12 times 12"],
                       "session_opening_ts": [10.0, 20.0]})
        self.assertEqual(bc._session_opening_turns,
                         ["Jarvis", "what's 12 times 12"])
        self.assertEqual(bc._session_opening_ts, [10.0, 20.0])
        # main() records the new turn first, exactly as it does live.
        bc._note_session_owner_utterance(self.Q)
        self.assertEqual(len(bc._session_opening_turns), 2)
        handled, _log = self._turn(self.Q)
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            'The first thing you asked me this session was: "what\'s 12 '
            'times 12", sir.')

    def test_a_payload_without_the_record_never_claims_nothing_was_asked(self):
        # The F6 repro: a swap FROM a version that did not carry the record.
        bc = self.bc
        self._consume({"conversation_tail": list(self.TAIL)})
        self.assertEqual(bc._session_opening_turns, [])
        bc._note_session_owner_utterance(self.Q)
        handled, _log = self._turn(self.Q)
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            "I no longer have the start of this session on record, sir.")
        # "what did I just ask you" still recalls the carried tail.
        self._speak.reset_mock()
        self._turn("what did I just ask you")
        self._speak.assert_called_once_with(
            'You asked me: "set a timer for ten minutes", sir.')
        # Second review (F5/F6 residual): one new ordinary turn used to fill
        # the record and become "the first thing". The start stays lost.
        self.assertEqual(bc._session_opening_lost, [True])
        bc._note_session_owner_utterance("turn on the desk lamp")
        bc.conversation_history.extend([
            {"role": "user", "content": "turn on the desk lamp"},
            {"role": "assistant", "content": "Done, sir."}])
        self._speak.reset_mock()
        self._turn(self.Q)
        self._speak.assert_called_once_with(
            "I no longer have the start of this session on record, sir.")

    def test_a_sender_that_lost_the_start_hands_that_on(self):
        bc = self.bc
        self._consume({"conversation_tail": list(self.TAIL),
                       "session_opening_turns": [],
                       "session_opening_ts": [],
                       "session_opening_lost": True})
        self.assertEqual(bc._session_opening_lost, [True])
        bc._note_session_owner_utterance("turn on the desk lamp")
        self.assertEqual(bc._session_opening_turns, [])

    def test_a_fresh_sender_with_no_turns_yet_loses_nothing(self):
        # No owner turn before the swap: the first one after it IS the first.
        bc = self.bc
        self._consume({"conversation_tail": [
                           {"role": "assistant", "content": "Good evening."}],
                       "session_opening_turns": [],
                       "session_opening_ts": []})
        self.assertEqual(bc._session_opening_lost, [False])
        bc._note_session_owner_utterance("what's 12 times 12")
        self._turn("what did I ask you first")
        self._speak.assert_called_once_with(
            'The first thing you asked me this session was: "what\'s 12 '
            'times 12", sir.')

    def test_the_seed_is_validated(self):
        bc = self.bc
        turns = ["Jarvis", 42, None, "  "] + [f"q{i}" for i in range(12)]
        self._consume({"session_opening_turns": turns,
                       "session_opening_ts": ["bad"]})
        # Strings only, the newest _SESSION_OPENING_MAX kept, a bad stamp
        # counts as now.
        self.assertEqual(len(bc._session_opening_turns),
                         bc._SESSION_OPENING_MAX)
        self.assertEqual(bc._session_opening_turns[-1], "q11")
        self.assertEqual(len(bc._session_opening_ts),
                         bc._SESSION_OPENING_MAX)
        for bad in ("not a list", {"a": 1}, None):
            with self.subTest(bad=bad):
                bc._session_opening_turns[:] = []
                bc._session_opening_ts[:] = []
                self._consume({"session_opening_turns": bad})
                self.assertEqual(bc._session_opening_turns, [])
        # Never overwrites a record this process already has.
        bc._session_opening_turns[:] = ["what's 12 times 12"]
        bc._session_opening_ts[:] = [1.0]
        self.assertEqual(bc._seed_session_opening(["other"], [2.0]), 0)
        self.assertEqual(bc._session_opening_turns, ["what's 12 times 12"])


if __name__ == "__main__":
    unittest.main()
