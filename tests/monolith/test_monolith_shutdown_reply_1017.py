"""Answering the overnight question in wake-word mode (live 2026-10-17 21:38).

"Jarvis, go ahead and shut down." armed the question "Would you like to start
the overnight protocol first, sir? Yes or no." The answer "Jarvis, no" was
transcribed "Jarvis Jarvis now": the prompt read "now" as unrelated, said
"Shutdown cancelled." and handed it to the brain, which re-ran an earlier web
search; the next bare "no" replies were thrown away by the wake-word gate.

Pinned here: a one-word no sound-alike shuts down; a one- or two-word reply
that is neither yes nor no is asked ONCE more (a longer one is still a real
request and goes through); and the question - and the re-ask - let the NEXT
utterance through the wake-word gate without the name, once.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


@requires_monolith
class ShutdownReplyTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        super().setUp()
        bc = self.bc
        orig = dict(bc._shutdown_prompt_pending)
        self.addCleanup(lambda: (bc._shutdown_prompt_pending.clear(),
                                 bc._shutdown_prompt_pending.update(orig)))
        admit = list(bc._standby_greet_admit_until)
        self.addCleanup(lambda: bc._standby_greet_admit_until.__setitem__(0, admit[0]))
        self.speak = mock.patch.object(bc, "_speak").start()
        self.addCleanup(mock.patch.stopall)
        self.shutdown = mock.patch.object(bc, "_act_shutdown_jarvis").start()
        self.overnight = mock.patch.object(bc, "_act_start_overnight_upgrade").start()

    def _arm(self):
        self.bc._shutdown_prompt_pending.update(
            armed=True, reasked=False, expires_at=time.time() + 100)

    def _spoken(self):
        return [c.args[0] for c in self.speak.call_args_list]

    # -- the misheard no -------------------------------------------------
    def test_the_live_misheard_no_shuts_down(self):
        self._arm()
        self.assertTrue(self.bc._handle_shutdown_prompt("Jarvis Jarvis now."))
        self.shutdown.assert_called_once()
        self.overnight.assert_not_called()
        self.assertNotIn("Shutdown cancelled.", self._spoken())

    def test_each_no_sound_alike_shuts_down(self):
        for said in ("Now.", "Know.", "Nah.", "Naw.", "Jarvis, now.", "No now."):
            with self.subTest(said=said):
                self.shutdown.reset_mock()
                self._arm()
                self.assertTrue(self.bc._handle_shutdown_prompt(said))
                self.shutdown.assert_called_once()

    # -- a garbled answer is asked once more ------------------------------
    def test_a_short_unclear_reply_is_asked_once_more(self):
        bc = self.bc
        self._arm()
        self.assertTrue(bc._handle_shutdown_prompt("Jarvis, what?"))
        self.shutdown.assert_not_called()
        self.overnight.assert_not_called()
        self.assertTrue(bc._shutdown_prompt_pending["armed"])
        self.assertIn("Yes or no", self._spoken()[-1])
        self.assertNotIn("Shutdown cancelled.", self._spoken())
        # ...and then a plain no answers it.
        self.assertTrue(bc._handle_shutdown_prompt("No."))
        self.shutdown.assert_called_once()

    def test_only_once_then_it_cancels(self):
        bc = self.bc
        self._arm()
        self.assertTrue(bc._handle_shutdown_prompt("um what"))
        self.assertFalse(bc._handle_shutdown_prompt("huh"))
        self.assertIn("Shutdown cancelled.", self._spoken())
        self.assertFalse(bc._shutdown_prompt_pending["armed"])
        self.shutdown.assert_not_called()

    def test_a_real_request_still_cancels_and_goes_through(self):
        self._arm()
        self.assertFalse(self.bc._handle_shutdown_prompt("what's the weather"))
        self.assertIn("Shutdown cancelled.", self._spoken())
        self.shutdown.assert_not_called()

    def test_a_hedged_yes_still_cancels(self):
        self._arm()
        self.assertFalse(self.bc._handle_shutdown_prompt("Yes, but later."))
        self.overnight.assert_not_called()
        self.shutdown.assert_not_called()

    # -- the reply needs no wake word, once --------------------------------
    def _gate(self, text):
        bc = self.bc
        with mock.patch.object(bc, "_require_wake_runtime", True), \
                mock.patch.object(bc._followup_window, "admit", return_value=False):
            return bc._should_refuse_background_audio(text)

    def test_the_question_lets_the_next_answer_through_without_the_name(self):
        bc = self.bc
        bc._standby_greet_admit_until[0] = 0.0
        self.assertTrue(bc._check_and_arm_shutdown_prompt("Jarvis, shut down."))
        refuse, _why = self._gate("no")
        self.assertFalse(refuse)                    # the bare "no" gets through
        refuse, why = self._gate("no")
        self.assertTrue(refuse)                     # once: the next one needs the name
        self.assertEqual(why, "wake-word mode")

    def test_the_re_ask_opens_it_again(self):
        bc = self.bc
        self._arm()
        bc._standby_greet_admit_until[0] = 0.0
        self.assertTrue(bc._handle_shutdown_prompt("Jarvis what"))
        refuse, _why = self._gate("no")
        self.assertFalse(refuse)


if __name__ == "__main__":
    unittest.main()
