"""Tests for the two render helpers in core.persona.

persona.py is mostly data, but render_signature_phrase_pool() and
render_tone_modulation_block() carry a documented FORMAT contract (they are
spliced verbatim into BASE_SYSTEM_PROMPT), so these tests pin that exact
shape — width-4 right-padded index + single-quoted phrase for the pool, and a
"  • <label> → <instruction>" bullet per tone in canonical order. A drift in
that formatting would silently corrupt the system prompt seam.

stdlib unittest only.
"""
from __future__ import annotations

import unittest

from core import persona


class SignaturePhrasePoolTests(unittest.TestCase):
    def setUp(self):
        self.rendered = persona.render_signature_phrase_pool()
        self.lines = self.rendered.split("\n")

    def test_one_line_per_phrase(self):
        self.assertEqual(len(self.lines), len(persona.JARVIS_SIGNATURE_PHRASES))

    def test_no_trailing_newline(self):
        # Docstring: trailing newline omitted; caller appends one.
        self.assertFalse(self.rendered.endswith("\n"))

    def test_first_line_format(self):
        first = persona.JARVIS_SIGNATURE_PHRASES[0]
        # Two leading spaces, "1." padded to width 4, then the quoted phrase.
        self.assertEqual(self.lines[0], f"  1.  '{first}'")

    def test_index_padding_aligns_single_and_double_digit(self):
        # "1." and "10." both pad to width 4 so the quote column lines up.
        self.assertTrue(self.lines[0].startswith("  1.  '"))
        self.assertTrue(self.lines[9].startswith("  10. '"))
        quote_col_1 = self.lines[0].index("'")
        quote_col_10 = self.lines[9].index("'")
        self.assertEqual(quote_col_1, quote_col_10)

    def test_phrases_are_single_quoted(self):
        for line, phrase in zip(self.lines, persona.JARVIS_SIGNATURE_PHRASES):
            self.assertTrue(line.endswith(f"'{phrase}'"))

    def test_indices_are_one_based_and_sequential(self):
        for i, line in enumerate(self.lines, start=1):
            self.assertIn(f"{i}.", line)


class ToneModulationBlockTests(unittest.TestCase):
    def setUp(self):
        self.rendered = persona.render_tone_modulation_block()
        self.lines = self.rendered.split("\n")

    def test_one_bullet_per_tone(self):
        self.assertEqual(len(self.lines), len(persona.TONE_MODULATION_RULES))

    def test_no_trailing_newline(self):
        self.assertFalse(self.rendered.endswith("\n"))

    def test_bullet_format_label_arrow_instruction(self):
        for line, rules in zip(self.lines, persona.TONE_MODULATION_RULES.values()):
            self.assertEqual(line, f"  • {rules['key_label']} → {rules['instruction']}")

    def test_canonical_order_preserved(self):
        # Order must follow the dict's insertion order.
        labels_in_order = [r["key_label"] for r in persona.TONE_MODULATION_RULES.values()]
        for line, label in zip(self.lines, labels_in_order):
            self.assertIn(label, line)

    def test_every_line_has_arrow(self):
        for line in self.lines:
            self.assertIn(" → ", line)


class OpenerFitRuleTests(unittest.TestCase):
    """2026-09-29 live: rotation pressure made the model open a joke and a
    dinner question with 'A bold choice, if I may say so, sir.' The prompt now
    binds the situational openers to their situation; these tests keep that
    rule in the prompt and keep every phrase it names inside the pool, so a
    later edit of the pool can't leave the rule pointing at nothing."""

    SITUATIONAL = ("A bold choice, if I may say so", "Are you quite sure",
                   "I'm afraid that's inadvisable", "I'm obliged to point out",
                   "Slight problem", "Rather concerning", "Shall I run the numbers",
                   "Based on current trajectory", "Projecting now")

    def setUp(self):
        from core import prompts
        self.prompt = prompts.BASE_SYSTEM_PROMPT
        self.rule = self.prompt[self.prompt.index("MATCH THE OPENER TO THE CONTENT"):]
        self.rule = self.rule[:self.rule.index("\n")]

    def test_rule_is_in_the_prompt_after_the_pool(self):
        self.assertLess(self.prompt.index(persona.render_signature_phrase_pool()),
                        self.prompt.index("MATCH THE OPENER TO THE CONTENT"))

    def test_every_situational_opener_named_is_in_the_pool(self):
        for phrase in self.SITUATIONAL:
            self.assertIn(phrase, self.rule)
            self.assertTrue(any(p.startswith(phrase) for p in persona.JARVIS_SIGNATURE_PHRASES),
                            f"{phrase!r} is no longer in the pool")

    def test_neutral_openers_named_are_in_the_pool(self):
        for phrase in ("Very good, sir.", "Certainly, sir.", "Of course, sir."):
            self.assertIn(phrase, self.rule)
            self.assertIn(phrase, persona.JARVIS_SIGNATURE_PHRASES)


class DeliverWhatWasAskedRuleTests(unittest.TestCase):
    """2026-09-29 live (v2.0.131): "tell me a short joke" got "I'm afraid I've
    run out of material, sir", and "tell me something interesting" got an
    invented remark about sir skipping dinner plus a search that never ran.
    The persona's "I'm afraid" / "taken the liberty" / pick-a-stance pressure
    beat the request. One short rule now binds open-ended requests to an
    actual answer; these tests keep it in the prompt, in the right place, and
    small (it sits in the cached prefix)."""

    def setUp(self):
        from core import prompts
        self.prompt = prompts.BASE_SYSTEM_PROMPT
        start = self.prompt.index("DELIVER WHAT WAS ASKED")
        self.rule = self.prompt[start:self.prompt.index("\n", start)]

    def test_rule_follows_the_opener_rule(self):
        self.assertLess(self.prompt.index("MATCH THE OPENER TO THE CONTENT"),
                        self.prompt.index("DELIVER WHAT WAS ASKED"))

    def test_rule_names_every_open_ended_request(self):
        for kind in ("joke", "fact", "story", "riddle", "recommendation",
                     "trivia"):
            self.assertIn(kind, self.rule)

    def test_rule_forbids_the_live_non_answers(self):
        low = self.rule.lower()
        self.assertIn("never decline", low)
        self.assertIn("run out", low)
        self.assertIn("quip about sir", low)
        self.assertIn("invent observations", low)
        self.assertIn("search", low)

    def test_rule_overrides_the_no_fun_facts_aside_rule(self):
        # The adjacent-fact rule says "never 'fun facts'"; the new rule must
        # say that applies to unrequested asides only, and precede it.
        self.assertIn("unrequested asides only", self.rule)
        self.assertLess(self.prompt.index("DELIVER WHAT WAS ASKED"),
                        self.prompt.index("never 'fun facts'"))

    def test_rule_is_one_short_line(self):
        self.assertLess(len(self.rule), 450)
        self.assertEqual(self.prompt.count("DELIVER WHAT WAS ASKED"), 1)


class NoEmptyPromiseExampleTests(unittest.TestCase):
    """v3.0.6. Live 2026-10-28 15:57:47: "I'll note that for posterity, sir."
    was spoken as a lead-in before a web_search and nothing was noted. The
    line was handed to the model as a stock example in five places of the
    base prompt, the signature pool and the MCU phrasebook (a stale
    duplicate), and the few-shot Example 3 also promised to "stand by to
    restore from backup". Every example the model is SHOWN must claim
    nothing that no action does."""

    import re as _re
    PROMISE_RE = _re.compile(
        r"for\s+posterity|i'?ll\s+note\s+(?:that|it)\b|"
        r"stand\s+by\s+to\s+restore", _re.IGNORECASE)

    def _shown(self):
        import mcu_phrases
        from core import prompts
        out = {
            "signature pool": persona.render_signature_phrase_pool(),
            "BASE_SYSTEM_PROMPT": prompts.BASE_SYSTEM_PROMPT,
            "BASE_SYSTEM_PROMPT_V2": prompts.BASE_SYSTEM_PROMPT_V2,
            "v3 swaps": "\n".join(new for _old, new in prompts._BASE_V3_SWAPS),
            "mcu phrasebook": "\n".join(
                line for bucket in mcu_phrases.MCU_PHRASES.values()
                for line in bucket),
        }
        # Both PROMPT_DIRECTIVE_V3 / PROMPT_INDEX_CANON states.
        v3 = prompts._apply_swaps(prompts.BASE_SYSTEM_PROMPT_V2,
                                  prompts._BASE_V3_SWAPS)
        out["v3 prompt"] = v3
        out["v3 canon prompt"] = prompts._strip_signature_block(v3)
        return out

    def test_no_shown_example_promises_a_note(self):
        for name, text in self._shown().items():
            with self.subTest(source=name):
                self.assertIsNone(self.PROMISE_RE.search(text),
                                  f"{name}: {self.PROMISE_RE.search(text)}")

    def test_every_v3_swap_still_applies(self):
        # The Example 3 / dry_wit literals changed with their swaps: a swap
        # whose old text drifted would be skipped silently.
        from core import prompts
        for old, _new in prompts._BASE_V3_SWAPS:
            with self.subTest(old=old[:50]):
                self.assertIn(old, prompts.BASE_SYSTEM_PROMPT_V2)

    def test_the_pool_keeps_its_shape(self):
        self.assertEqual(len(persona.JARVIS_SIGNATURE_PHRASES), 20)
        self.assertIn("One does admire the optimism, sir.",
                      persona.JARVIS_SIGNATURE_PHRASES)

    def test_the_retired_line_is_still_caught(self):
        # A model that learned the old line is still caught when it copies it
        # into a proactive remark (core/proactive_guard RETIRED_EXAMPLES).
        from core import proactive_guard as pg
        pg.persona_pool.cache_clear()
        self.addCleanup(pg.persona_pool.cache_clear)
        self.assertTrue(pg.persona_copy("I'll note that for posterity, sir."))


if __name__ == "__main__":
    unittest.main()
