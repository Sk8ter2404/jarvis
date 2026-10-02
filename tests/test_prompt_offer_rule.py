"""The adjacent-fact rule closes earlier turns' offers (NEW #13, 2026-10-01).

Live 2026-10-01 21:36: both replies to "how's my print doing?" ended "Also,
I've opened those search results in your browser; shall I read them to you
via screen vision?" - an offer from a chain that had ended 35 minutes
earlier, riding the "Also, sir..." adjacent-fact slot. The dispatcher drops
such a repeat (tests/monolith/test_monolith_stale_offers.py); the persona's
rule now says the same, in both prompt variants.

stdlib unittest only; light tier.
"""
from __future__ import annotations

import unittest

from core import prompts

_RULE = "an offer sir did not take up is closed"


class OfferRuleTests(unittest.TestCase):
    def test_the_adjacent_fact_rule_closes_old_offers(self):
        base = prompts.BASE_SYSTEM_PROMPT
        start = base.index("Adjacent-fact volunteering")
        rule = base[start:base.index("Examples:", start)]
        self.assertIn(_RULE, rule)
        self.assertIn("never re-raise an earlier turn's offer or question",
                      rule)

    def test_both_prompt_variants_carry_it(self):
        for night_quiet in (True, False):
            with self.subTest(night_quiet=night_quiet):
                self.assertIn(_RULE,
                              prompts.base_system_prompt(night_quiet))


if __name__ == "__main__":
    unittest.main()
