"""core.name_suggest - "did you mean Claude?" (2026-10-05).

Live 00:25:02 Parakeet heard "close everything except for Claude" as
"... except for Claw"; the bulk close kept nothing and closed nothing (right)
and offered nothing (the fix). The same day "Skrillex" came out as
"Skrillix", "Skrylix" and "Skrillig". Pure module, light tier.

    python -m unittest tests.test_name_suggest
"""
from __future__ import annotations

import unittest

from core import name_suggest as ns

# The names the live desktop's windows answered to (titles generic).
LIVE_NAMES = ["Claude", "Claude Code", "Google Chrome", "Chrome",
              "File Explorer", "Explorer", "Media Player", "Downloads"]


class SuggestTests(unittest.TestCase):
    def test_the_live_mishearing_suggests_claude(self):
        self.assertEqual(ns.suggest("Claw", LIVE_NAMES), "Claude")

    def test_one_form_in_two_candidates_is_not_a_tie(self):
        # "Claude" (the app) and "Claude Code" (a terminal) both offer the
        # word "Claude": one suggestion, not an ambiguity.
        self.assertEqual(ns.suggest("Claw", ["Claude Code", "Claude"]),
                         "Claude")

    def test_sound_alike_and_spelling_slips(self):
        for heard, want in (("Clod", "Claude"), ("Crome", "Chrome"),
                            ("Explorr", "Explorer"), ("spotfy", "Spotify"),
                            ("notpad", "Notepad")):
            with self.subTest(heard=heard):
                self.assertEqual(
                    ns.suggest(heard, LIVE_NAMES + ["Spotify", "Notepad"]),
                    want)

    def test_nothing_close_is_no_suggestion(self):
        for heard in ("Spotify", "garage door", "lights", "Netflix"):
            with self.subTest(heard=heard):
                self.assertIsNone(ns.suggest(heard, LIVE_NAMES))

    def test_a_name_that_is_open_gets_no_other_name(self):
        self.assertIsNone(ns.suggest("Claude", LIVE_NAMES))
        self.assertIsNone(ns.suggest("file explorer", LIVE_NAMES))

    def test_a_tie_between_two_different_names_is_no_guess(self):
        self.assertIsNone(ns.suggest("Clam", ["Claim", "Clamp"]))

    def test_short_or_empty_names_never_match(self):
        self.assertIsNone(ns.suggest("Cl", LIVE_NAMES))
        self.assertIsNone(ns.suggest("", LIVE_NAMES))
        self.assertIsNone(ns.suggest(None, LIVE_NAMES))
        self.assertIsNone(ns.suggest("Claw", []))
        self.assertIsNone(ns.suggest("Claw", None))

    def test_long_titles_are_compared_by_their_words(self):
        title = "A very long document title that nobody would ever say aloud"
        self.assertEqual(ns.suggest("documant", [title]), "document")

    def test_never_raises(self):
        self.assertIsNone(ns.suggest(object(), [object(), 3, None]))
        self.assertEqual(ns.score(object(), None), 0.0)


class PhoneticKeyTests(unittest.TestCase):
    def test_the_skrillex_family_sounds_alike(self):
        key = ns.phonetic_key("Skrillex")
        for heard in ("Skrillix", "Skrylix"):
            with self.subTest(heard=heard):
                self.assertEqual(ns.phonetic_key(heard), key)
        # "acrylics" is a different word to the ear and to the key.
        self.assertNotEqual(ns.phonetic_key("acrylics"), key)

    def test_spelling_variants_score_as_suggestions(self):
        for heard in ("Skrillix", "Skrylix", "Skrillig"):
            with self.subTest(heard=heard):
                self.assertGreater(ns.score(heard, "Skrillex"), 0.0)
        self.assertEqual(ns.score("acrylics", "Skrillex"), 0.0)

    def test_empty(self):
        self.assertEqual(ns.phonetic_key(""), "")
        self.assertEqual(ns.phonetic_key(None), "")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
