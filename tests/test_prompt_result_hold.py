"""The result-hold prompt rule ships on every route (NEW #1, 2026-10-01).

Live 2026-10-01 the local model wrote "[ACTION: see_screen, …] I've reviewed
the search results, sir; …" - describing a result it had not seen. The
dispatcher no longer speaks prose after such a token
(bobert_companion._result_hold_cut, tests/monolith/test_monolith_result_hold);
this rule tells the model to stop there. It lives inside
PC_CONTROL_SAFETY_RULES, the block every route ships (the full PC prompt, the
prompt router's always-shipped core, the local cheatsheet's tail).

stdlib unittest only; light tier.
"""
from __future__ import annotations

import unittest

from core import prompts


class ResultHoldRuleTests(unittest.TestCase):
    def test_rule_is_in_the_safety_block(self):
        rules = prompts.PC_CONTROL_SAFETY_RULES
        self.assertIn("End your reply at its token", rules)
        self.assertIn("never describe, guess or summarise what it will find",
                      rules)
        self.assertIn("never say you have read or reviewed it", rules)

    def test_rule_is_single_sourced(self):
        self.assertIn(prompts.RESULT_HOLD_RULE,
                      prompts.PC_CONTROL_SAFETY_RULES)
        self.assertEqual(prompts.PC_CONTROL_PROMPT.count(
            prompts.RESULT_HOLD_RULE), 1)

    def test_rule_names_the_look_up_actions_of_the_live_turns(self):
        for name in ("see_screen", "web_search", "check_print"):
            self.assertIn(name, prompts.RESULT_HOLD_RULE)


if __name__ == "__main__":
    unittest.main()
