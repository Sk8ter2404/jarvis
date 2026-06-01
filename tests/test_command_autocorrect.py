"""Logic tests for command_autocorrect.py — the fuzzy action-name corrector
the dispatcher consults when the LLM emits an action name that isn't a literal
key in ACTIONS.

Embeddings are forced OFF for every test (disable_embeddings) so the suite is
deterministic and never touches the Ollama endpoint. The lexical scoring path
(Levenshtein + token overlap + phonetic boost) is what the dispatcher falls
back to whenever Ollama is down anyway, so it's the load-bearing surface.

All scores asserted below were observed from the module directly with
embeddings disabled; they pin the corrected-name decisions and the
threshold/ambiguity contract, not just types.
"""
from __future__ import annotations

import unittest

import command_autocorrect as ac


# Representative slice of the real ACTIONS registry the dispatcher passes in.
ACTIONS = [
    "screenshot", "open_url", "launch_app", "shutdown_jarvis",
    "focus_mode", "night_owl_mode", "see_screen",
    "ambient_mode", "ambient_mode_on", "ambient_listening",
    "set_timer", "play_music", "volume_up", "volume_down",
]


class EmbeddingsOffMixin:
    """Force the embedding pathway off for the whole class and restore after."""

    @classmethod
    def setUpClass(cls):
        ac.disable_embeddings()

    @classmethod
    def tearDownClass(cls):
        ac.enable_embeddings()


# ── Normalisation / tokenisation (pure) ──────────────────────────────────────
class NormaliseTests(unittest.TestCase):
    def test_separators_become_underscore(self):
        self.assertEqual(ac._normalise("Night-Owl Mode"), "night_owl_mode")
        self.assertEqual(ac._normalise("a/b.c"), "a_b_c")

    def test_collapse_and_strip(self):
        self.assertEqual(ac._normalise("  __Focus  Mode__  "), "focus_mode")

    def test_tokens_drop_filler(self):
        # 'turn', 'on', 'mode' are all filler → only 'ambient' survives.
        self.assertEqual(ac._tokens("turn_on_ambient_mode"), ["ambient"])

    def test_tokens_keep_originals_when_all_filler(self):
        # Everything is filler → fall back to the raw tokens rather than [].
        self.assertEqual(ac._tokens("the_a_an"), ["the", "a", "an"])


# ── Edit-distance primitives (pure) ──────────────────────────────────────────
class LevenshteinTests(unittest.TestCase):
    def test_classic_distance(self):
        self.assertEqual(ac._levenshtein("kitten", "sitting"), 3)
        self.assertEqual(ac._levenshtein("", "abc"), 3)
        self.assertEqual(ac._levenshtein("abc", "abc"), 0)

    def test_ratio_bounds(self):
        self.assertEqual(ac._lev_ratio("abc", "abc"), 1.0)
        self.assertEqual(ac._lev_ratio("", ""), 1.0)
        # 3 edits over a max length of 7 → 1 - 3/7.
        self.assertAlmostEqual(ac._lev_ratio("kitten", "sitting"), 1.0 - 3 / 7)


# ── autocorrect_command: exact / near-miss / below-threshold ─────────────────
class AutocorrectCommandTests(EmbeddingsOffMixin, unittest.TestCase):
    def _best(self, unknown, acts=ACTIONS, threshold=0.75):
        return ac.autocorrect_command(unknown, acts, threshold,
                                      use_embeddings=False)

    def test_exact_match_scores_one(self):
        self.assertEqual(self._best("screenshot"), ("screenshot", 1.0))

    def test_case_and_whitespace_normalised_to_exact(self):
        # Mixed case + spaces normalise to the same key → still a 1.0 match.
        self.assertEqual(self._best("  Focus Mode  "), ("focus_mode", 1.0))
        self.assertEqual(self._best("SHUTDOWN_JARVIS"), ("shutdown_jarvis", 1.0))
        self.assertEqual(self._best("launch app"), ("launch_app", 1.0))

    def test_near_miss_typo_is_corrected(self):
        # A single-separator typo lands well above threshold and maps to the
        # intended action.
        name, score = self._best("screen_shot")
        self.assertEqual(name, "screenshot")
        self.assertGreaterEqual(score, 0.75)

    def test_transposition_is_corrected(self):
        name, score = self._best("screenshto")
        self.assertEqual(name, "screenshot")
        self.assertGreaterEqual(score, 0.75)

    def test_phonetic_misspelling_corrected(self):
        # 'nite' → 'night' is a phonetic/edit near-miss, still resolves.
        name, score = self._best("nite_owl_mode")
        self.assertEqual(name, "night_owl_mode")
        self.assertGreaterEqual(score, 0.75)

    def test_unknown_token_left_uncorrected(self):
        # A genuinely unrelated token must NOT be corrected — returns
        # (None, best_seen) so the caller can log the near-miss.
        name, score = self._best("totally_unrelated_xyzzy")
        self.assertIsNone(name)
        self.assertLess(score, 0.75)

    def test_too_far_token_below_threshold_not_corrected(self):
        # 'scren shot' (dropped 'e' AND split) scores ~0.67 — deliberately
        # below the 0.75 floor, so it is left uncorrected. Pinning this guards
        # the confidence threshold against being loosened by accident.
        name, score = self._best("scren shot")
        self.assertIsNone(name)
        self.assertLess(score, 0.75)
        self.assertGreater(score, 0.5)  # still the best-seen, just not enough

    def test_threshold_is_respected_as_boundary(self):
        # Same token clears a lowered threshold but not the default — proves
        # the threshold argument actually gates the decision.
        self.assertIsNone(self._best("scren shot", threshold=0.75)[0])
        self.assertEqual(self._best("scren shot", threshold=0.60)[0], "screenshot")

    def test_empty_unknown_returns_none(self):
        self.assertEqual(self._best(""), (None, 0.0))

    def test_empty_and_none_candidates_are_skipped(self):
        # The 'if not cand: continue' guard must skip ''/None without crashing.
        self.assertEqual(
            ac.autocorrect_command("opn_url", [None, "open_url", ""],
                                   use_embeddings=False)[0],
            "open_url",
        )
        self.assertEqual(
            ac.autocorrect_command("screenshot", [None, ""],
                                   use_embeddings=False),
            (None, 0.0),
        )

    def test_alias_is_the_same_function(self):
        self.assertIs(ac.autocorrect, ac.autocorrect_command)


# ── autocorrect_command_topk ─────────────────────────────────────────────────
class TopKTests(EmbeddingsOffMixin, unittest.TestCase):
    def test_returns_k_sorted_descending(self):
        top = ac.autocorrect_command_topk("ambient", ACTIONS, k=3,
                                          use_embeddings=False)
        self.assertEqual(len(top), 3)
        scores = [s for _, s in top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_threshold_applied(self):
        # Even a poor unknown yields up to k candidates (caller inspects gap).
        top = ac.autocorrect_command_topk("zzz_qqq", ACTIONS, k=2,
                                          use_embeddings=False)
        self.assertEqual(len(top), 2)

    def test_k_zero_and_empty_unknown(self):
        self.assertEqual(ac.autocorrect_command_topk("x", ACTIONS, k=0), [])
        self.assertEqual(ac.autocorrect_command_topk("", ACTIONS, k=2), [])


# ── autocorrect_command_choice: silent / ambiguous / none ────────────────────
class ChoiceTests(EmbeddingsOffMixin, unittest.TestCase):
    def test_silent_when_clear_winner(self):
        # 'screenshto' resolves to screenshot far above any runner-up.
        choice = ac.autocorrect_command_choice("screenshto", ACTIONS,
                                               use_embeddings=False)
        self.assertEqual(choice["status"], "silent")
        self.assertEqual(choice["primary"][0], "screenshot")
        self.assertIsNone(choice["secondary"])

    def test_ambiguous_when_two_tie_within_gap(self):
        # 'ambient' scores identically against ambient_mode and
        # ambient_mode_on (both reduce to the single token 'ambient'),
        # so the resolver must ask rather than silently pick one.
        acts = ["ambient_mode", "ambient_mode_on", "screenshot", "open_url"]
        choice = ac.autocorrect_command_choice("ambient", acts,
                                               use_embeddings=False)
        self.assertEqual(choice["status"], "ambiguous")
        self.assertIsNotNone(choice["secondary"])
        names = {choice["primary"][0], choice["secondary"][0]}
        self.assertEqual(names, {"ambient_mode", "ambient_mode_on"})
        # Both clear the threshold and the gap is within the default 0.10.
        self.assertGreaterEqual(choice["primary"][1], 0.75)
        self.assertGreaterEqual(choice["secondary"][1], 0.75)
        self.assertLessEqual(
            choice["primary"][1] - choice["secondary"][1], 0.10)

    def test_none_when_nothing_clears_threshold(self):
        choice = ac.autocorrect_command_choice("zzz_qqq_unmatched", ACTIONS,
                                               use_embeddings=False)
        self.assertEqual(choice["status"], "none")
        # primary still set to best-seen so caller can log it.
        self.assertIsNotNone(choice["primary"])
        self.assertLess(choice["primary"][1], 0.75)
        self.assertIsNone(choice["secondary"])

    def test_empty_registry_is_none_with_null_primary(self):
        choice = ac.autocorrect_command_choice("anything", [],
                                               use_embeddings=False)
        self.assertEqual(choice["status"], "none")
        self.assertIsNone(choice["primary"])
        self.assertIsNone(choice["secondary"])


# ── Embedding availability toggle (no network involved) ──────────────────────
class EmbeddingToggleTests(unittest.TestCase):
    def tearDown(self):
        ac.enable_embeddings()  # leave the module in its default state

    def test_disable_then_enable_flips_availability(self):
        ac.disable_embeddings()
        self.assertFalse(ac._embeddings_available())
        ac.enable_embeddings()
        self.assertTrue(ac._embeddings_available())

    def test_score_identical_normalised_names_is_one(self):
        # 'Open URL' and 'open_url' normalise equal → exact 1.0, no network.
        ac.disable_embeddings()
        self.assertEqual(ac._score("Open URL", "open_url",
                                   use_embeddings=False), 1.0)


if __name__ == "__main__":
    unittest.main()
