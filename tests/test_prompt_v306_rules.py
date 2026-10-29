"""v3.0.6 prompt rules from the 2026-10-28 17:53-18:10 human-style live test.

Each rule is read through the REAL router entry the live local turn uses
(core.prompt_router.slim_pc_control for the stable core, turn_pc_block for
the routed sections of that utterance), with the live utterance as the
fixture:

  * F1 "set a timer for three minutes" became set_timer "3 minutes | check
    the oven" - the model copied the only example. The example is now bare
    and says no message unless he gave one;
  * F2 "what's a good quick dinner idea?" ran web_search. Ideas,
    suggestions and facts the model knows are answered in words (BASIC
    web_search entry, in the stable core every turn);
  * F7 "did you send that email" ran list_pending_drafts ("No pending
    drafts") although no email was ever asked for: say plainly none was
    asked for or sent, no token (EMAIL TRIAGE, routed on "email").

Each assertion fails on the v3.0.5 text (62d520b), which had none of these
lines and carried "[ACTION: set_timer, 5 minutes | check the oven]".
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import prompt_router as pr  # noqa: E402
from core import prompts  # noqa: E402

# Both PROMPT_ROUTED_V3 states: the pre-swap text and the text this process
# ships.
_PCS = {"v2": prompts.PC_CONTROL_PROMPT_V2, "live": prompts.PC_CONTROL_PROMPT}


class TimerExampleTests(unittest.TestCase):
    UTTERANCE = "set a timer for three minutes"

    def test_the_example_is_bare_and_says_no_invented_message(self):
        for name, pc in _PCS.items():
            with self.subTest(prompt=name):
                block = pr.turn_pc_block(self.UTTERANCE, pc)
                self.assertIn("TIMERS / REMINDERS", block)
                self.assertIn("[ACTION: set_timer, 3\n    minutes]", block)
                self.assertIn("No | message unless he gave one; never "
                              "invent one.", block)

    def test_the_copied_label_is_no_longer_an_example_token(self):
        for name, pc in _PCS.items():
            with self.subTest(prompt=name):
                self.assertNotIn("[ACTION: set_timer, 5 minutes | check the "
                                 "oven]", pc)


class IdeasInWordsTests(unittest.TestCase):
    UTTERANCE = "what's a good quick dinner idea?"

    def test_the_stable_core_says_ideas_are_answered_in_words(self):
        for name, pc in _PCS.items():
            with self.subTest(prompt=name):
                core = pr.slim_pc_control(self.UTTERANCE, pc)
                at = core.index("web_search, <query>")
                entry = core[at:core.index("youtube, <query>", at)]
                self.assertIn("Ideas, suggestions, facts you", entry)
                self.assertIn("('a dinner idea') = words, no token.", entry)


class DidYouSendThatEmailTests(unittest.TestCase):
    UTTERANCE = "did you send that email"

    def test_the_email_section_says_none_was_sent_and_no_token(self):
        for name, pc in _PCS.items():
            with self.subTest(prompt=name):
                block = pr.turn_pc_block(self.UTTERANCE, pc)
                self.assertIn("EMAIL TRIAGE", block)
                self.assertIn("'Did you send that email?' when no email was "
                              "asked for or sent in this\n    conversation: "
                              "say plainly none was asked for or sent. No "
                              "token.", block)


if __name__ == "__main__":
    unittest.main()
