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


if __name__ == "__main__":
    unittest.main()
