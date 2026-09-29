"""mcu_phrases: the rotation hint is separate from the phrasebook (2026-09-29).

The "last used" hint changes almost every turn, so it must not be part of the
system prompt (it changed the local model's cached prefix on every rotation).
render_phrasebook_block() with no argument is the constant block the system
prompt carries; render_rotation_hint() is the per-turn text.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import mcu_phrases as mp  # noqa: E402


class RotationHintTests(unittest.TestCase):
    def test_phrasebook_without_argument_carries_no_hint(self):
        block = mp.render_phrasebook_block()
        self.assertNotIn("last used", block)
        self.assertEqual(block, mp.render_phrasebook_block({}))

    def test_hint_names_each_used_line_in_bucket_order(self):
        hint = mp.render_rotation_hint(
            {"minimal": "Working.", "acknowledgements": "Very good, sir."})
        self.assertIn("acknowledgements: 'Very good, sir.'", hint)
        self.assertIn("minimal: 'Working.'", hint)
        self.assertLess(hint.index("acknowledgements"), hint.index("minimal"))
        self.assertIn("different", hint)

    def test_hint_is_deterministic(self):
        a = mp.render_rotation_hint({"status": "Running the numbers now.",
                                     "concern": "x"})
        b = mp.render_rotation_hint({"concern": "x",
                                     "status": "Running the numbers now."})
        self.assertEqual(a, b)

    def test_empty_or_bad_input_is_empty(self):
        for bad in (None, {}, "greeting", ["x"], {"acknowledgements": ""},
                    {"not_an_intent": "Quite."}, {"minimal": 3}):
            with self.subTest(bad=bad):
                self.assertEqual(mp.render_rotation_hint(bad), "")


if __name__ == "__main__":
    unittest.main()
