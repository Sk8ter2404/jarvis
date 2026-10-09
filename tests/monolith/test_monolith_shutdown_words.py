"""The owner's four shutdown tries (live 2026-10-06 22:28-22:30), through the
REAL routers in the main loop's order.

  1. "Jarvis shut down."        armed the overnight prompt.
  2. "Jarvis shut down no overnight protocol."
                                the prompt called it UNRELATED, cancelled, and
                                the model then said it was powering down with
                                nothing run.
  3. "Jarvis shut down with no overnight protocol."
                                7 words - past the pre-router; the model's
                                shutdown_jarvis was held: "with" read as an
                                object.
  4. "Jarvis, yes, that is what I want."
                                "[confirm] unrelated reply" - the held
                                shutdown was cancelled.

Now (2) and (3) shut JARVIS down without the overnight protocol, and (4)
confirms the held action. The speech that is not the owner asking - "don't
shut down", "should I shut down my PC?", "shut down the printer", a quoted
line, a broadcast sentence, "No, don't", a yes to a DIFFERENT question -
still never shuts JARVIS down.

ReviewRoutingTests (review 2026-10-09) adds what the first cut got wrong: a
device statement or a question that shut JARVIS down or armed the prompt, a
change of mind ("No, stay on.") read as a no, a bare "That's what I said."
answering a different question - and the owner's phrasings it still missed
("Yes, shut down with no overnight protocol.", "Skip the overnight
protocol.", "Yes, run the overnight protocol, then shut down.").

    python -m unittest tests.monolith.test_monolith_shutdown_words
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

SAID_1 = "Jarvis shut down."
SAID_2 = "Jarvis shut down no overnight protocol."
SAID_3 = "Jarvis shut down with no overnight protocol."
SAID_4 = "Jarvis, yes, that is what I want."


class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        bc = self.bc
        self.spoken = []
        self._p(bc, "_speak", lambda text, *a, **k: self.spoken.append(text))
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        self._p(bc, "PC_CONTROL_ENABLED", True)
        self._p(bc._cmd_autocorrect, "_embed_disabled", True)
        # Every way JARVIS can go: the pre-router's direct calls and the
        # dispatcher's ACTIONS entries are the same two mocks.
        self.shutdown = mock.Mock(return_value="Goodbye, sir.")
        self.overnight = mock.Mock(return_value="Standing by, sir.")
        self.wiped = mock.Mock(return_value="wiped")
        self._p(bc, "_act_shutdown_jarvis", self.shutdown)
        self._p(bc, "_act_start_overnight_upgrade", self.overnight)
        self.acts = {
            "shutdown_jarvis": self.shutdown,
            "exit_jarvis": self.shutdown,
            "start_overnight_upgrade": self.overnight,
            "wipe_thing_x": self.wiped,
            "get_time": mock.Mock(return_value="noon"),
        }
        self._p(bc, "ACTIONS", self.acts)
        self._p(bc, "_shutdown_prompt_pending",
                {"armed": False, "expires_at": 0.0})
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_confirmation_at", [0.0])
        self._p(bc, "_pending_autocorrect_choice", [])

    def _route(self, text):
        """The main loop's pre-routers, in its order (bobert_companion's
        voice loop: shutdown prompt, shutdown trigger, autocorrect pick,
        pending confirmation). The name of the one that consumed ``text``,
        or None - it would go on to the model."""
        bc = self.bc
        for router in (bc._handle_shutdown_prompt,
                       bc._check_and_arm_shutdown_prompt,
                       bc.handle_autocorrect_disambig_response,
                       bc.handle_confirmation_response):
            if router(text):
                return router.__name__
        return None

    def _arm(self):
        bc = self.bc
        bc._shutdown_prompt_pending["armed"] = True
        bc._shutdown_prompt_pending["expires_at"] = time.time() + 30

    def _dispatch(self, reply, user_text):
        bc = self.bc
        prev = bc._begin_turn_grounding(user_text)
        try:
            return bc.parse_and_run_actions(reply)
        finally:
            bc._end_turn_grounding(prev)

    def _hold_shutdown(self):
        """The model's shutdown_jarvis for words that did not ask for it is
        held for a yes - the state the owner's fourth try answered."""
        self._dispatch("Right away, sir. [ACTION: shutdown_jarvis]",
                       user_text="Jarvis, turn it off.")
        self.shutdown.assert_not_called()
        self.assertEqual(self.bc._pending_confirmation,
                         [("shutdown_jarvis", "")])
        self.spoken.clear()


@requires_monolith
class OwnersFourTriesTests(_Base):
    def test_the_live_sequence_shuts_down_on_the_second_try(self):
        bc = self.bc
        self.assertEqual(self._route(SAID_1), "_check_and_arm_shutdown_prompt")
        self.assertTrue(bc._shutdown_prompt_pending["armed"])
        self.shutdown.assert_not_called()
        self.assertEqual(self._route(SAID_2), "_handle_shutdown_prompt")
        self.shutdown.assert_called_once()
        self.overnight.assert_not_called()
        self.assertNotIn("Shutdown cancelled.", self.spoken)

    def test_try_two_and_its_variants_answer_the_prompt_with_no(self):
        for said in (SAID_2, SAID_3,
                     "Jarvis, shut down, no overnight protocol.",
                     "shut down without the overnight protocol",
                     "Jarvis, without the protocol.",
                     "Without the overnight protocol, please.",
                     "Jarvis, with no overnight protocol.",
                     "No overnight protocol, just shut down.",
                     "Power down without the overnight protocol."):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.overnight.reset_mock()
                self._arm()
                self.assertEqual(self._route(said), "_handle_shutdown_prompt")
                self.shutdown.assert_called_once()
                self.overnight.assert_not_called()
                self.assertFalse(self.bc._shutdown_prompt_pending["armed"])

    def test_try_three_with_no_prompt_open_shuts_down_at_once(self):
        # Live, the prompt had been cancelled by try 2, so try 3 met no
        # open question - and its 7 words skipped the pre-router.
        for said in (SAID_3, SAID_2,
                     "Jarvis, shut down now with no overnight protocol.",
                     "Hey Jarvis, power down without the overnight protocol.",
                     "Jarvis, no overnight protocol, just shut down."):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.overnight.reset_mock()
                self.bc._shutdown_prompt_pending["armed"] = False
                self.assertEqual(self._route(said),
                                 "_check_and_arm_shutdown_prompt")
                self.shutdown.assert_called_once()
                self.overnight.assert_not_called()
                self.assertFalse(self.bc._shutdown_prompt_pending["armed"])
                self.assertFalse(any("overnight protocol first" in s
                                     for s in self.spoken), self.spoken)

    def test_try_three_through_the_model_is_not_held(self):
        for said in (SAID_3, SAID_2, "Jarvis, shut down without the protocol.",
                     "Jarvis, shut down for the night.",
                     "Jarvis, shut down right now."):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.bc._pending_confirmation.clear()
                self._dispatch("Right away, sir. [ACTION: shutdown_jarvis]",
                               user_text=said)
                self.shutdown.assert_called_once()
                self.assertEqual(self.bc._pending_confirmation, [])

    def test_try_four_confirms_the_held_shutdown(self):
        for said in (SAID_4, "Jarvis, yes please do.", "Jarvis, yes, do it.",
                     "Yes, that's what I want.", "Jarvis, yes, that is what "
                     "I said."):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.bc._pending_confirmation.clear()
                self._hold_shutdown()
                self.assertEqual(self._route(said),
                                 "handle_confirmation_response")
                self.shutdown.assert_called_once()
                self.assertNotIn("Cancelled, sir.", self.spoken)


@requires_monolith
class NotTheOwnerAskingTests(_Base):
    COLD = (
        "Jarvis, don't shut down.",
        "Don't shut down",
        "Jarvis, do not shut down with no overnight protocol.",
        "Never go offline.",
        "Don't turn yourself off.",
        "should I shut down my PC?",
        "Should I shut down?",
        "Jarvis, shut down the printer.",
        "shut down the robot",
        "Jarvis, shut down the browser.",
        "Jarvis, shut down the printer with no overnight protocol.",
        'He said "shut down."',
        'The sign said "shut down with no overnight protocol."',
        # broadcast / video lines
        "They shut down with no warning.",
        "The factory had to shut down with no warning.",
        "Officials shut down the highway without the permit.",
        "It shut down last night.",
    )

    def test_cold_speech_never_shuts_down_or_arms_the_prompt(self):
        for said in self.COLD:
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.overnight.reset_mock()
                self.bc._shutdown_prompt_pending["armed"] = False
                self._route(said)
                self.assertFalse(self.bc._shutdown_prompt_pending["armed"])
                # ...so a plain "No." after it has nothing to confirm.
                self.assertIsNone(self._route("No."))
                self.shutdown.assert_not_called()
                self.overnight.assert_not_called()

    def test_refusing_the_open_prompt_never_shuts_down(self):
        for said in ("No, don't shut down.", "Jarvis, no, don't.",
                     "Don't shut down.", "Jarvis, don't shut down.",
                     "No, wait.",
                     "No, don't shut down, no overnight protocol.",
                     "Shut down the printer with no overnight protocol."):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.overnight.reset_mock()
                self.spoken.clear()
                self._arm()
                self.assertIsNone(self._route(said))
                self.shutdown.assert_not_called()
                self.overnight.assert_not_called()
                self.assertIn("Shutdown cancelled.", self.spoken)
                self.assertFalse(self.bc._shutdown_prompt_pending["armed"])

    def test_refusing_the_held_shutdown_cancels_it(self):
        for said in ("Jarvis, no, don't.", "No.", "Jarvis, don't shut down."):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.bc._pending_confirmation.clear()
                self.bc._shutdown_prompt_pending["armed"] = False
                self._hold_shutdown()
                self._route(said)
                self.shutdown.assert_not_called()
                self.assertEqual(self.bc._pending_confirmation, [])

    def test_the_model_cannot_shut_down_on_words_that_did_not_ask(self):
        for said in ("Jarvis, don't shut down.", "Should I shut down?",
                     "Jarvis, shut down the printer with no overnight "
                     "protocol.", 'He said "shut down now."'):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self.bc._pending_confirmation.clear()
                self._dispatch("Right away, sir. [ACTION: shutdown_jarvis]",
                               user_text=said)
                self.shutdown.assert_not_called()
                self.assertEqual(self.bc._pending_confirmation,
                                 [("shutdown_jarvis", "")])

    def test_a_yes_to_a_different_question_answers_that_question(self):
        bc = self.bc
        # A pending delete: the owner's yes runs IT, not a shutdown.
        bc._queue_pending_confirmation("wipe_thing_x", "all")
        self.assertEqual(self._route(SAID_4), "handle_confirmation_response")
        self.wiped.assert_called_once_with("all")
        self.shutdown.assert_not_called()
        # The overnight question: yes is the overnight protocol, not a
        # full power-off.
        self._arm()
        self.assertEqual(self._route(SAID_4), "_handle_shutdown_prompt")
        self.overnight.assert_called_once()
        self.shutdown.assert_not_called()
        # Nothing pending: the yes goes to the model, nothing runs.
        self.assertIsNone(self._route(SAID_4))
        self.shutdown.assert_not_called()


@requires_monolith
class ReviewRoutingTests(_Base):
    """Review 2026-10-09, through the real routers. Contexts: "A" the
    overnight question is open, "B" nothing is pending, "C" the model's
    shutdown_jarvis is held for a yes, "D" a different action (a wipe) is
    held."""

    def _end_state(self, ctx, *seq):
        bc = self.bc
        self.shutdown.reset_mock()
        self.overnight.reset_mock()
        self.wiped.reset_mock()
        self.spoken.clear()
        bc._shutdown_prompt_pending["armed"] = False
        bc._pending_confirmation.clear()
        bc._pending_confirmation_at[0] = 0.0
        if ctx == "A":
            self._arm()
        elif ctx == "C":
            bc._queue_pending_confirmation("shutdown_jarvis", "")
        elif ctx == "D":
            bc._queue_pending_confirmation("wipe_thing_x", "all")
        consumed = None
        for said in seq:
            consumed = self._route(said)
        if self.shutdown.called:
            return "SHUTDOWN"
        if self.overnight.called:
            return "OVERNIGHT"
        if self.wiped.called:
            return "WIPE"
        if bc._shutdown_prompt_pending["armed"]:
            return "ASKS"
        if any("ancelled" in s for s in self.spoken):
            return "CANCEL"
        return "LLM" if consumed is None else "CONSUMED"

    def _expect(self, ctx, outcome, cases):
        for seq in cases:
            seq = (seq,) if isinstance(seq, str) else tuple(seq)
            with self.subTest(ctx=ctx, seq=seq):
                self.assertEqual(self._end_state(ctx, *seq), outcome)

    def test_the_owners_four_tries_still_work(self):
        self.assertEqual(self._end_state("B", SAID_1), "ASKS")
        self.assertEqual(self._end_state("B", SAID_1, SAID_2), "SHUTDOWN")
        self.assertEqual(self._end_state("B", SAID_3), "SHUTDOWN")
        self.assertEqual(self._end_state("C", SAID_4), "SHUTDOWN")

    def test_a_change_of_mind_cancels_the_prompt(self):
        self._expect("A", "CANCEL", (
            "No, stay on.", "No, keep running.", "No, I changed my mind.",
            "Nope, stay awake.", "No no, keep going.", "No, I need you.",
            "Nope, scratch that.", "No, forget it.",
            "No, I don't want you to shut down.", "Cancel the shutdown.",
            "No, no shutdown.", "Jarvis, no updates on the printer?",
            "Just shut down the printer.", "No, the overnight protocol."))

    def test_a_device_statement_then_a_correction_never_shuts_down(self):
        for seq in (("Jarvis, laptop shut down without updating.", "No."),
                    ("Server shut down without the update.", "No."),
                    ("My laptop shut down without updating.",
                     "No, my laptop."),
                    ("The printer shut down without updating.",
                     "No, the printer did."),
                    ("It'll shut down without updating.", "No, the PC."),
                    ("I'll shut down without the update.",
                     "No, I meant my PC."),
                    ('Jarvis, shut down "Plex".', "No, Plex."),
                    ("Cancel the shutdown.", "No."),
                    ("Jarvis, no shutdown.", "No."),
                    ("My laptop shut down.", "No."),
                    ("The factory had to shut down.", "No."),
                    ("Windows will shut down.", "No."),
                    ("The TV will power off.", "No.")):
            with self.subTest(seq=seq):
                self.assertNotIn(self._end_state("B", *seq),
                                 ("SHUTDOWN", "OVERNIGHT"))

    def test_questions_echoes_and_quotes_with_the_prompt_open_cancel(self):
        self._expect("A", "CANCEL", (
            "What happens if you shut down without the overnight protocol?",
            "Is it okay to shut down without the update?",
            "Shut down with no overnight protocol? No, wait.",
            "Shut down. No, overnight protocol.",
            '"Shut down without the protocol."',
            "That's what I said.", "Jarvis, that's what I asked."))

    def test_update_words_and_questions_never_shut_down_at_once(self):
        self._expect("B", "LLM", (
            "Shut down? No updates?", "Shut down. Not updating.",
            "Jarvis, power off, no updates.",
            "Jarvis, shut down without updating?",
            "Jarvis, shut down without updating.",
            "Jarvis, shut down. No, overnight protocol."))

    def test_the_overnight_protocol_by_name_starts_it(self):
        self._expect("A", "OVERNIGHT", (
            "Yes, run the overnight protocol, then shut down.",
            "Overnight protocol, then shut down.",
            "Do the overnight protocol and then shut down.",
            "Start the overnight protocol and shut down.",
            "Yes, overnight protocol.", "Yeah, do the overnight protocol.",
            "Run the overnight protocol.", "Jarvis yes do the overnight",
            "Jarvis, shut down with the overnight protocol.",
            "Jarvis, Jarvis, yes.", "Um, yes."))

    def test_more_ways_to_say_no_overnight_with_the_prompt_open(self):
        self._expect("A", "SHUTDOWN", (
            "Jarvis shut down but no overnight protocol.",
            "Jarvis shut down, skip the overnight protocol.",
            "Skip the overnight protocol.",
            "Jarvis shut down, don't do the overnight protocol.",
            "I don't want the overnight protocol.",
            "No, don't bother with the overnight protocol.",
            "Jarvis, shut it down, no overnight protocol.",
            "Jarvis shut down, no over night protocol.",
            "Jarvis, no, don't do overnight, shut down.",
            "Jarvis turn off, no overnight protocol.",
            "Negative.", "Um, no.", "Jarvis, Jarvis, no.", "No, I'm good.",
            "No. Shut down without the overnight protocol."))

    def test_more_ways_shut_down_at_once_with_nothing_open(self):
        self._expect("B", "SHUTDOWN", (
            "Jarvis, Jarvis, shut down with no overnight protocol.",
            "Jarvis, shut down without running the overnight protocol.",
            "Jarvis shut down, don't run the overnight protocol.",
            "Jarvis, power off, don't do the overnight protocol.",
            "Jarvis, shut down, skip the overnight protocol.",
            "Jarvis, shut it down, no overnight protocol.",
            "Jarvis shut down, no over night protocol."))

    def test_the_held_shutdown_takes_a_restated_yes(self):
        self._expect("C", "SHUTDOWN", (
            "Yes, shut down with no overnight protocol.",
            "Jarvis, yes, shut down with no overnight protocol.",
            "Yes, that's exactly what I want.", "Yes, I meant that.",
            "I said yes.", "Yes, you heard me.", "Just do it.",
            "Please, just do it.", "Jarvis, Jarvis, yes.", "Um, yes.",
            "Yes, shut it down.", "Yes, I want you to shut down.",
            "Yes, power down.", "Yes, shut down.", "Yes, shut down now.",
            "Yeah, I'm sure, shut down.",
            "Yes, shut down, no overnight protocol.",
            "That is what I want."))

    def test_the_held_shutdown_still_refuses(self):
        for said in ("Jarvis, no, don't.", "No.", "Jarvis, don't shut down.",
                     "Yes, shut down later.", "Yes, shut down the printer.",
                     "That's what she said.", "Please.", 'Yes, "shut down".',
                     "Yes, shut down?"):
            with self.subTest(said=said):
                self.assertNotIn(self._end_state("C", said),
                                 ("SHUTDOWN", "OVERNIGHT"))

    def test_a_restatement_never_answers_a_different_question(self):
        for said in ("That's what I said.", "Jarvis, that's what I meant.",
                     "That's what I asked", "That is what I want."):
            with self.subTest(said=said):
                self.assertEqual(self._end_state("D", said), "CANCEL")
                self.wiped.assert_not_called()
                self.shutdown.assert_not_called()
        # ...while a yes to it still runs it, not a shutdown.
        self.assertEqual(self._end_state("D", "Yes, that's what I said."),
                         "WIPE")
        self.shutdown.assert_not_called()


@requires_monolith
class PoweringDownClaimIsNeverSpokenTests(_Base):
    def test_the_live_reply_is_withheld_for_a_follow_up(self):
        cleaned, results = self._dispatch(
            "Very good, sir. I've cancelled the overnight protocol and am "
            "powering down now.", user_text=SAID_2)
        self.shutdown.assert_not_called()
        self.assertEqual(cleaned, "")
        self.assertEqual([r[0] for r in results], ["_unverified_claim"])


if __name__ == "__main__":
    unittest.main()
