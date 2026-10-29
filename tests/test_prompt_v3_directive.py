"""Prompt v3 core (v3 overhaul W03): PROMPT_DIRECTIVE_V3.

One bool setting swaps a handful of prompt passages (report 06, E1-E3 and
E5-E10):

  * OFF (the default before 3.0.0's fair prompt gate): the previous text,
    byte for byte - core/prompts.py is only ever changed by exact swaps and
    the old directive survives as _LOCAL_MODE_DIRECTIVE_V2;
  * ON: an [ACTION] token is a REQUEST (rule 3 loses its "unless you emitted
    its [ACTION" wording), rule 2 says when to SEARCH, rule 6 names
    how_is_the_print, the few-shot examples carry tokens, "no way to check"
    waits for the ACTION INDEX, and the never-guess guard has its blank line.

The pure-text tests need no monolith; the directive / guard tests do.
"""
from __future__ import annotations

import hashlib
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import prompts  # noqa: E402
from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)

# sha256 of _LOCAL_MODE_DIRECTIVE as shipped in v2.0.231.
_DIRECTIVE_V2_SHA = (
    "4e4bf15c255458dfae3047a41db6f4fd1a46ed75a7bbc484012d6c0756cd3853")


class SwapTableTests(unittest.TestCase):
    """The v3 swaps in core/prompts.py find their text and read as intended."""

    def test_every_swap_finds_its_old_text(self):
        for old, _new in prompts._BASE_V3_SWAPS:
            self.assertIn(old, prompts.BASE_SYSTEM_PROMPT_V2, old[:60])
        for old, _new in prompts._SAFETY_V3_SWAPS:
            self.assertIn(old, prompts.PC_CONTROL_SAFETY_RULES_V2, old[:60])

    def test_off_path_is_the_untouched_text(self):
        # With the switch off the exported constants ARE the old text.
        with mock.patch.object(prompts, "_prompt_v3_on", return_value=False):
            self.assertEqual(
                prompts._apply_swaps("abc", ()), "abc")
        if not prompts._prompt_v3_on() and not prompts._canon_on():
            self.assertIs(prompts.BASE_SYSTEM_PROMPT,
                          prompts.BASE_SYSTEM_PROMPT_V2)
            self.assertIs(prompts.PC_CONTROL_SAFETY_RULES,
                          prompts.PC_CONTROL_SAFETY_RULES_V2)

    def _v3_base(self):
        return prompts._apply_swaps(prompts.BASE_SYSTEM_PROMPT_V2,
                                    prompts._BASE_V3_SWAPS)

    def test_swaps_change_only_what_they_name(self):
        v3 = self._v3_base()
        self.assertNotEqual(v3, prompts.BASE_SYSTEM_PROMPT_V2)
        # Putting every old fragment back gives the old text exactly.
        back = v3
        for old, new in prompts._BASE_V3_SWAPS:
            self.assertIn(new, back)
            back = back.replace(new, old, 1)
        self.assertEqual(back, prompts.BASE_SYSTEM_PROMPT_V2)

    def test_stance_one_no_longer_teaches_done_before_the_result(self):
        v3 = self._v3_base()
        old = "task acknowledged or done. Open with an acknowledgement"
        self.assertIn(old, prompts.BASE_SYSTEM_PROMPT_V2)
        self.assertNotIn(old, v3)
        self.assertIn("'Done, sir.' only in a reply with no token", v3)

    def test_few_shot_header_and_order_survive(self):
        v3 = self._v3_base()
        head = ("Few-shot examples of the dry-wit register — match this "
                "calibration, not louder, not flatter:")
        self.assertIn(head, v3)
        self.assertIn(head, prompts.BASE_SYSTEM_PROMPT_V2)
        self.assertLess(v3.index(head), v3.index("Memory use: when you know"))
        # The bodies carry tokens, and no "restarting now" without one.
        self.assertIn("[ACTION: launch_app, Fortnite]", v3)
        self.assertIn("[ACTION: system_pulse]", v3)
        self.assertIn("[ACTION: play_music, Michael Jackson]", v3)
        shots = v3[v3.index(head):v3.index("Memory use: when you know")]
        self.assertNotIn("restarting now", shots)
        self.assertNotIn("stand by to restore from backup", shots)
        # A question never sits beside a token in an example.
        self.assertNotIn("quite sure", shots)

    def test_opener_rule_stays_one_line_with_every_phrase(self):
        v3 = self._v3_base()
        start = v3.index("MATCH THE OPENER TO THE CONTENT")
        rule = v3[start:v3.index("\n", start)]
        for phrase in ("A bold choice, if I may say so", "Are you quite sure",
                       "I'm afraid that's inadvisable", "I'm obliged to point out",
                       "Slight problem", "Rather concerning",
                       "Shall I run the numbers", "Based on current trajectory",
                       "Projecting now"):
            self.assertIn(phrase, rule)
        self.assertIn("QUESTIONS for sir", rule)
        self.assertIn("beside an [ACTION] token", rule)

    def test_opener_rule_never_argues_against_ask_first(self):
        """Security review PROMPT-3 (2026-10-27): the persona line must not
        contradict the kept SAFETY rule 'If unsure ..., ASK FIRST'."""
        v3 = self._v3_base()
        self.assertNotIn("never ask permission for what you can simply do", v3)
        safety = prompts._apply_swaps(prompts.PC_CONTROL_SAFETY_RULES_V2,
                                      prompts._SAFETY_V3_SWAPS)
        self.assertIn("ASK FIRST", safety)
        with mock.patch.object(prompts, "_prompt_v3_on", return_value=True):
            self.assertNotIn("never ask permission for what you can simply do",
                             prompts.base_system_prompt())

    def test_intent_tag_lines(self):
        v3 = self._v3_base()
        self.assertIn("only when sir announces a risky choice", v3)
        # Reviewed 2026-10-28: the replacement snark line carried no rule.
        self.assertIn("'One does admire the optimism, sir.' on the same "
                      "terms: only for a plainly over-hopeful plan", v3)
        self.assertIn("Use ONLY the ten words above", v3)
        self.assertNotIn("Unrecognised tags also fall back", v3)

    def test_signature_pool_untouched(self):
        from core import persona
        v3 = self._v3_base()
        for phrase in persona.JARVIS_SIGNATURE_PHRASES:
            self.assertIn(phrase, v3)

    def test_no_way_to_check_is_gated_on_the_index(self):
        v3 = prompts._apply_swaps(prompts.PC_CONTROL_SAFETY_RULES_V2,
                                  prompts._SAFETY_V3_SWAPS)
        self.assertIn("ONLY after looking down the action list", v3)
        self.assertNotIn("ACTION INDEX", v3)  # test_monolith_prompt_slim pins it
        # The quoted sentence is pinned verbatim by other suites.
        self.assertIn("I'm afraid I've no way to check that, sir.", v3)
        self.assertEqual(v3.count("ASK FIRST"),
                         prompts.PC_CONTROL_SAFETY_RULES_V2.count("ASK FIRST"))
        # The confirmation-hold paragraph is not touched.
        head = prompts.PC_CONTROL_SAFETY_RULES_V2.split("\n\n", 1)[0]
        self.assertTrue(v3.startswith(head))


@requires_monolith
class DirectiveTests(MonolithGlobalsTestCase):
    def test_old_directive_is_kept_byte_for_byte(self):
        old = self.bc._LOCAL_MODE_DIRECTIVE_V2
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),
                         _DIRECTIVE_V2_SHA)
        self.assertIn("unless you emitted its [ACTION", old)

    def test_the_live_directive_follows_the_setting(self):
        bc = self.bc
        want = (bc._LOCAL_MODE_DIRECTIVE_V3
                if getattr(bc, "PROMPT_DIRECTIVE_V3", False)
                else bc._LOCAL_MODE_DIRECTIVE_V2)
        self.assertEqual(bc._LOCAL_MODE_DIRECTIVE, want)

    def test_default_is_on_in_the_shipped_config(self):
        # Default ON since 3.0.0 (fair prompt gate, PLAN.md: golden 90.1% -> 94.0%).
        import core.config as cfg
        self.assertIn("PROMPT_DIRECTIVE_V3 = True",
                      open(cfg.__file__, encoding="utf-8").read())

    def test_v3_rule_three_is_a_request_not_a_result(self):
        v3 = self.bc._LOCAL_MODE_DIRECTIVE_V3
        old = self.bc._LOCAL_MODE_DIRECTIVE_V2
        self.assertTrue(v3.startswith(
            "\n\n----------\nYOU ARE RUNNING ON THE LOCAL MODEL."))
        self.assertNotIn("unless you emitted its [ACTION", v3)
        self.assertIn("is a REQUEST, not a result", v3)
        self.assertIn("write NOTHING after it", v3)
        self.assertIn("look down the index first", v3)
        self.assertIn("SEARCH ([ACTION: web_search, ...]", v3)
        self.assertIn("[ACTION: how_is_the_print]", v3)
        self.assertNotIn("[ACTION: print_status]", v3)
        # Arithmetic / banter still answer with no token.
        self.assertIn("arithmetic, facts, definitions, opinions, banter",
                      v3)
        self.assertIn("NO token", v3)
        # Rules 1, 4, 5 and 7 are untouched.
        for keep in ("1. To DO anything on the computer", "4. Be CONCISE",
                     "5. Do NOT lecture", "7. Stay in character as JARVIS"):
            self.assertIn(keep, v3)
            self.assertIn(keep, old)
        self.assertTrue(v3.endswith("----------\n"))

    def test_every_directive_swap_applies(self):
        bc = self.bc
        for old, _new in bc._DIRECTIVE_V3_SWAPS:
            self.assertIn(old, bc._LOCAL_MODE_DIRECTIVE_V2)

    def test_guard_header_gets_its_blank_line_only_when_on(self):
        bc = self.bc
        guard = bc._LOCAL_NEVER_GUESS_GUARD
        for flag, prefix in ((False, ""), (True, "\n\n")):
            with mock.patch.object(bc, "PROMPT_DIRECTIVE_V3", flag,
                                   create=True):
                bc._SLIM_PC_CACHE.clear()
                try:
                    out = bc._slim_stable_pc("", None)
                finally:
                    bc._SLIM_PC_CACHE.clear()
            self.assertTrue(out.startswith(prefix + guard), flag)
            self.assertFalse(out.startswith("\n\n\n"), flag)

    def test_setting_is_a_turn_flag_key(self):
        self.assertIn("PROMPT_DIRECTIVE_V3", self.bc._TURN_FLAG_KEYS)


if __name__ == "__main__":
    unittest.main()
