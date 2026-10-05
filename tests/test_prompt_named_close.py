"""The prompt a named close reaches the brain with (2026-10-05).

Live 00:24:21-00:25:43 the owner said "close out File Explorer", "close file
explorer" (twice) and "close Google Chrome"; the brain answered every one with
[ACTION: close_last_opened] (only what JARVIS itself opened) and all four
failed. Root cause: those turns name no "window", so the router shipped only
the always-on MULTI-MONITOR APP LAUNCHING section - and the ONLY close token
in it was close_last_opened ("close that"). close_window lived in WINDOW
MANAGEMENT, which never loaded. One turn earlier "close that WINDOW for me,
media player" loaded it and the brain got close_window right.

Pinned: on every one of those turns the prompt carries close_window for a
NAMED close, and says close_last_opened is only for a close with no name; the
sign-in rule rides the core preamble every route ships. Light tier.

    python -m unittest tests.test_prompt_named_close
"""
from __future__ import annotations

import re
import unittest

from core import prompt_router as pr
from core.prompts import PC_CONTROL_PROMPT, PC_CONTROL_SAFETY_RULES

LIVE = (
    "Jarvis, go ahead and close out File Explorer 2.",
    "Jarvis, close file explorer.",
    "Jarvis Close File Explorer.",
    "Jarvis close Google Chrome.",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


class NamedCloseRoutingTests(unittest.TestCase):
    def test_every_live_turn_ships_close_window(self):
        for said in LIVE:
            with self.subTest(said=said):
                slim = pr.slim_pc_control(said, PC_CONTROL_PROMPT)
                self.assertIn("close_window, <name>", slim)
                self.assertIn("[ACTION: close_window, File Explorer]",
                              _flat(slim))

    def test_close_last_opened_is_documented_as_the_nameless_close(self):
        slim = _flat(pr.slim_pc_control(LIVE[1], PC_CONTROL_PROMPT))
        m = re.search(r"close_last_opened\s+—\s+(.{0,120})", slim)
        self.assertIsNotNone(m)
        self.assertIn("ONLY a close with NO name", m.group(1))

    def test_it_lives_in_the_cache_stable_half(self):
        # The always-on section is hoisted into the byte-stable block, so the
        # line costs the KV cache nothing turn to turn.
        stable = _flat(pr.stable_pc_block(PC_CONTROL_PROMPT))
        self.assertIn("[ACTION: close_window, File Explorer]", stable)
        for said in LIVE:
            with self.subTest(said=said):
                self.assertNotIn("close_window, <name>",
                                 pr.turn_pc_block(said, PC_CONTROL_PROMPT))


class SignInRuleTests(unittest.TestCase):
    def test_the_sign_in_rule_is_in_the_safety_rules(self):
        rules = _flat(PC_CONTROL_SAFETY_RULES)
        self.assertIn("A sign-in, account-chooser, consent or password page "
                      "is sir's to complete", rules)
        self.assertIn("unless he asks for that exact click this turn", rules)

    def test_every_route_ships_it(self):
        core, _sections = pr.split_pc_control(PC_CONTROL_PROMPT)
        self.assertIn("account-chooser", core)
        slim = pr.slim_pc_control("pull up that page so I can sign in",
                                  PC_CONTROL_PROMPT)
        self.assertIn("account-chooser", slim)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
