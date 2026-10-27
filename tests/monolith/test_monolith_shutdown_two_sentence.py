"""v3 W08 through the REAL routers: "Jarvis, shut down. No overnight
protocol." (live 2026-10-19 20:26:04, replay c044) went to the model, which
claimed "I've initiated the shutdown"; the self-terminate hold caught it and
the owner's later "yes" came after the hold lapsed, so nothing happened.

Now the two-sentence form (a shutdown of JARVIS + only an overnight clause)
arms the existing overnight prompt and asks its existing question; the
existing yes/no handler answers the next reply. It never shuts down on its
own. SHUTDOWN_TWO_SENTENCE_ASK = False is exactly the old routing.

    python -m unittest tests.monolith.test_monolith_shutdown_two_sentence
"""
from __future__ import annotations

import time
import unittest

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_shutdown_words import _Base

QUESTION = ("Would you like to start the overnight protocol first, sir? "
            "Yes or no.")
SAID_1019 = "Jarvis, shut down. No overnight protocol."   # replay c044
TYPED_1019 = "shut down. No overnight protocol."


@requires_monolith
class TwoSentenceShutdownTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "SHUTDOWN_TWO_SENTENCE_ASK", True)

    def _fresh(self):
        bc = self.bc
        self.shutdown.reset_mock()
        self.overnight.reset_mock()
        self.spoken.clear()
        bc._shutdown_prompt_pending.clear()
        bc._shutdown_prompt_pending.update({"armed": False,
                                            "expires_at": 0.0})
        bc._pending_confirmation.clear()

    def test_the_1019_phrase_arms_and_asks_never_reaching_the_model(self):
        for said in (SAID_1019, TYPED_1019,
                     "Jarvis, shut down. No, overnight protocol."):
            with self.subTest(said=said):
                self._fresh()
                self.assertEqual(self._route(said),
                                 "_check_and_arm_shutdown_prompt")
                self.assertTrue(self.bc._shutdown_prompt_pending["armed"])
                self.assertEqual(self.spoken, [QUESTION])
                self.shutdown.assert_not_called()
                self.overnight.assert_not_called()

    def test_no_afterwards_shuts_down_through_the_existing_path(self):
        for said in (SAID_1019, "Jarvis, shut down. No, overnight protocol."):
            with self.subTest(said=said):
                self._fresh()
                self._route(said)
                self.assertEqual(self._route("No."),
                                 "_handle_shutdown_prompt")
                self.shutdown.assert_called_once()
                self.overnight.assert_not_called()
                self.assertFalse(self.bc._shutdown_prompt_pending["armed"])

    def test_yes_afterwards_starts_the_overnight_protocol(self):
        self._fresh()
        self._route("Jarvis, shut down. No, overnight protocol.")
        self.assertEqual(self._route("Yes."), "_handle_shutdown_prompt")
        self.overnight.assert_called_once()
        self.shutdown.assert_not_called()

    def test_the_lookalikes_never_arm(self):
        for said in ("Shut down with no overnight protocol? No, wait.",
                     "Shut down chrome. No overnight protocol.",
                     "Jarvis. Shut down. No overnight protocol.",
                     "Shut down. No overnight protocol. Thanks.",
                     "What happens if you shut down. No overnight protocol.",
                     "Don't shut down. No overnight protocol.",
                     "My laptop shut down. No overnight protocol.",
                     'Shut down "Plex". No overnight protocol.',
                     "Shut down. No updates."):
            with self.subTest(said=said):
                self._fresh()
                self._route(said)
                self.assertFalse(self.bc._shutdown_prompt_pending["armed"])
                self.assertNotIn(QUESTION, self.spoken)
                self.shutdown.assert_not_called()
                self.overnight.assert_not_called()

    def test_an_answer_to_the_open_prompt_is_not_re_asked(self):
        # The prompt's own handler answers (here: cancels) it, exactly as
        # before; the pre-router does not arm it again in the same breath.
        self._fresh()
        self._arm()
        self.assertIsNone(self._route("Shut down. No, overnight protocol."))
        self.assertFalse(self.bc._shutdown_prompt_pending["armed"])
        self.assertIn("Shutdown cancelled.", self.spoken)
        self.assertNotIn(QUESTION, self.spoken)
        self.shutdown.assert_not_called()

    def test_an_expired_prompt_asks_again(self):
        self._fresh()
        self.bc._shutdown_prompt_pending.update(
            {"armed": True, "expires_at": time.time() - 1})
        self.assertEqual(self._route(SAID_1019),
                         "_check_and_arm_shutdown_prompt")
        self.assertTrue(self.bc._shutdown_prompt_pending["armed"])
        self.assertEqual(self.spoken, [QUESTION])

    def test_the_one_sentence_forms_are_unchanged(self):
        self._fresh()
        self.assertEqual(self._route("Jarvis shut down."),
                         "_check_and_arm_shutdown_prompt")
        self.assertEqual(self.spoken, [QUESTION])
        self._fresh()
        self.assertEqual(
            self._route("Jarvis shut down with no overnight protocol."),
            "_check_and_arm_shutdown_prompt")
        self.shutdown.assert_called_once()
        self.assertNotIn(QUESTION, self.spoken)

    def test_a_held_shutdown_is_replaced_by_the_question(self):
        # Review 2026-10-27: the model's shutdown held for a yes stayed
        # queued, so "Yes." started the overnight protocol and a second
        # "Yes." ran the stale shutdown_jarvis.
        for said in (SAID_1019, "Jarvis, shut down. No, overnight protocol."):
            with self.subTest(said=said):
                self._fresh()
                self._hold_shutdown()
                self.assertEqual(self._route(said),
                                 "_check_and_arm_shutdown_prompt")
                self.assertEqual(self.bc._pending_confirmation, [])
                self.assertEqual(self.spoken, [QUESTION])
                self.assertEqual(self._route("Yes."),
                                 "_handle_shutdown_prompt")
                self.overnight.assert_called_once()
                self._route("Yes.")
                self.shutdown.assert_not_called()
                self.overnight.assert_called_once()
                self.assertEqual(self.bc._pending_confirmation, [])

    def test_the_setting_is_logged_with_the_turn_flags(self):
        self.assertIn("SHUTDOWN_TWO_SENTENCE_ASK", self.bc._TURN_FLAG_KEYS)


@requires_monolith
class SettingOffTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "SHUTDOWN_TWO_SENTENCE_ASK", False)

    def test_off_is_the_old_routing_the_phrase_goes_to_the_model(self):
        for said in (SAID_1019, TYPED_1019,
                     "Jarvis, shut down. No, overnight protocol."):
            with self.subTest(said=said):
                self.spoken.clear()
                self.assertIsNone(self._route(said))
                self.assertFalse(self.bc._shutdown_prompt_pending["armed"])
                self.assertEqual(self.spoken, [])
                self.shutdown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
