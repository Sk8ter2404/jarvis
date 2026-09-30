"""Tests for core/device_speech_filter.py's EXPECTED LINES / DIALOGUE API
(expect / expect_done / forget_expected / begin_dialogue / end_dialogue /
dialogue_active / dialogue_source, and match()'s expected-line rules).

A skill that makes a device speak a line it composed at run time registers
the line first, so JARVIS's mic does not take the device for the owner. The
window is SHORT (the predicted end + 4 s, hard 30 s) and the bars are the
phrase files' own: the owner's reply a few seconds later must always get
through.

GENERIC fixtures only (a made-up "desk device"). No phrase files: the data dir
is pointed at an empty temp dir so only expected lines can match.

stdlib unittest only; CI-safe.
    python -m unittest tests.test_device_speech_filter_expect
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from core import device_speech_filter as dsf

_SRC = "desk device"
_LINE = "I think the fridge respects me deeply."


class _Base(unittest.TestCase):
    def setUp(self):
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        self.tmp = tempfile.mkdtemp(prefix="dsf_expect_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.tmp})
        env.start()
        self.addCleanup(env.stop)

    def expect(self, text=_LINE, window_s=3.0, now=100.0, source=_SRC):
        return dsf.expect(source, text, window_s=window_s, now=now)


class PublicConstantsTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(dsf.EXPECT_MAX, 32)
        self.assertEqual(dsf.EXPECT_GRACE_S, 4.0)
        self.assertEqual(dsf.EXPECT_HARD_TTL_S, 30.0)
        self.assertEqual(dsf.EXPECTED_FRAGMENT_MIN_WORDS, 4)
        self.assertEqual(dsf.EXPECTED_FRAGMENT_MIN_RATIO, 0.85)

    def test_stop_words_public_alias(self):
        self.assertIs(dsf.STOP_WORDS, dsf._STOP_WORDS)
        for w in ("stop", "halt", "freeze", "abort", "emergency", "estop",
                  "cancel"):
            self.assertIn(w, dsf.STOP_WORDS)


class ExpectMatchingTests(_Base):
    def test_exact_line_matches_with_source_and_normalised_phrase(self):
        self.expect()
        hit = dsf.match("I think the fridge respects me deeply", now=101.0)
        self.assertEqual(hit, (_SRC, "i think the fridge respects me deeply",
                               1.0))

    def test_no_expectation_no_match(self):
        self.assertIsNone(dsf.match(_LINE, now=101.0))

    def test_whole_ratio_long_line_uses_the_080_bar(self):
        self.expect("The kettle wants a word with you.")
        hit = dsf.match("the cattle want a word with him", now=101.0)
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit[2], 0.80)
        self.assertLess(hit[2], 0.85)

    def test_short_line_needs_090(self):
        # 4 words / 16 chars = short: 0.875 is not enough.
        self.expect("The lamp is warm.")
        self.assertIsNone(dsf.match("the lamp is worn", now=101.0))
        self.assertIsNotNone(dsf.match("the lamp is warm", now=101.0))

    def test_two_word_line_is_exact_only(self):
        self.expect("Four socks.")
        self.assertIsNone(dsf.match("for socks", now=101.0))
        self.assertIsNotNone(dsf.match("four socks", now=101.0))

    def test_four_word_fragment_matches(self):
        self.expect()
        hit = dsf.match("fridge respects me deeply", now=101.0)
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit[2], dsf.EXPECTED_FRAGMENT_MIN_RATIO)

    def test_three_word_fragment_does_not_match(self):
        self.expect()
        self.assertIsNone(dsf.match("respects me deeply", now=101.0))
        self.assertIsNone(dsf.match("the fridge respects", now=101.0))

    def test_owner_quote_inside_a_longer_sentence_does_not_match(self):
        self.expect()
        self.assertIsNone(dsf.match(
            "I think the fridge respects me deeply, which is nice",
            now=101.0))

    def test_stop_word_never_matches(self):
        self.expect("Please stop poking my buttons now.")
        self.assertIsNone(dsf.match("please stop poking my buttons now",
                                    now=101.0))

    def test_protected_owner_phrase_never_matches(self):
        self.expect("Yes.")
        self.assertIsNone(dsf.match("yes", now=101.0))
        self.expect("Hey there, desk.")
        self.assertIsNone(dsf.match("hey there desk", now=101.0,
                                    never_match={"hey there desk"}))

    def test_no_relax_during_a_dialogue(self):
        tok = dsf.begin_dialogue(_SRC, now=100.0)
        self.assertTrue(tok)
        self.expect("The lamp is warm.")
        self.assertIsNone(dsf.match("the lamp is worn", now=101.0))


class ExpectWindowTests(_Base):
    def test_owner_paraphrase_filtered_1s_after_done_but_not_5s(self):
        h = self.expect(window_s=3.0, now=100.0)
        dsf.expect_done(h, now=103.0)
        paraphrase = "i think the fridge respects me too"
        self.assertIsNotNone(dsf.match(paraphrase, now=104.0))
        # A fresh registration (the check above is non-destructive, but the
        # window test below must see its own entry).
        dsf._reset_cache_for_tests()
        h = self.expect(window_s=3.0, now=200.0)
        dsf.expect_done(h, now=203.0)
        self.assertIsNone(dsf.match(paraphrase, now=208.0))

    def test_window_expires_without_done(self):
        self.expect(window_s=3.0, now=100.0)
        self.assertIsNotNone(dsf.match(_LINE, now=106.9))
        self.assertIsNone(dsf.match(_LINE, now=107.1))

    def test_hard_ttl_caps_a_long_window(self):
        self.expect(window_s=500.0, now=100.0)
        self.assertIsNotNone(dsf.match(_LINE, now=129.0))
        self.assertIsNone(dsf.match(_LINE, now=130.5))

    def test_done_late_is_still_capped_by_the_hard_ttl(self):
        h = self.expect(window_s=1.0, now=100.0)
        dsf.expect_done(h, now=128.0)     # after its own window expired
        self.assertIsNone(dsf.match(_LINE, now=129.0))
        h = self.expect(window_s=25.0, now=200.0)     # expires 229
        dsf.expect_done(h, now=228.0)                 # 232, capped at 230
        self.assertIsNotNone(dsf.match(_LINE, now=229.5))
        self.assertIsNone(dsf.match(_LINE, now=230.5))

    def test_done_early_shortens_the_window(self):
        h = self.expect(window_s=10.0, now=100.0)
        dsf.expect_done(h, now=101.0)
        self.assertIsNone(dsf.match(_LINE, now=105.5))

    def test_expect_max_drops_the_oldest(self):
        first = self.expect("Line number zero is the oldest one.", now=100.0)
        for i in range(dsf.EXPECT_MAX):
            self.expect(f"Filler line {'x' * (i + 1)} for the bound.",
                        now=100.0)
        self.assertIsNotNone(first)
        self.assertIsNone(dsf.match("line number zero is the oldest one",
                                    now=100.5))
        self.assertEqual(len(dsf._expected_snapshot(100.5)), dsf.EXPECT_MAX)

    def test_expect_rejects_empty_nonprintable_and_blank_source(self):
        self.assertIsNone(dsf.expect(_SRC, "", window_s=1.0))
        self.assertIsNone(dsf.expect(_SRC, "  ", window_s=1.0))
        self.assertIsNone(dsf.expect(_SRC, "bad\x07bell", window_s=1.0))
        self.assertIsNone(dsf.expect("", "a fine line", window_s=1.0))
        self.assertIsNone(dsf.expect(_SRC, None, window_s=1.0))
        self.assertIsNotNone(dsf.expect(_SRC, "a fine line", window_s=1.0))

    def test_forget_expected(self):
        self.expect(now=100.0)
        self.expect("Another line from somewhere else entirely.", now=100.0,
                    source="other")
        self.assertEqual(dsf.forget_expected(_SRC), 1)
        self.assertIsNone(dsf.match(_LINE, now=101.0))
        self.assertIsNotNone(dsf.match(
            "another line from somewhere else entirely", now=101.0))
        self.assertEqual(dsf.forget_expected(), 1)
        self.assertEqual(dsf.forget_expected(), 0)

    def test_unknown_handle_is_ignored(self):
        dsf.expect_done(12345, now=1.0)
        dsf.expect_done(None, now=1.0)


class DialogueBracketTests(_Base):
    def test_active_from_begin_through_the_tail(self):
        self.assertFalse(dsf.dialogue_active(now=99.0))
        tok = dsf.begin_dialogue(_SRC, max_s=60.0, now=100.0)
        self.assertTrue(dsf.dialogue_active(now=100.0))
        self.assertEqual(dsf.dialogue_source(now=100.0), _SRC)
        dsf.end_dialogue(tok, tail_s=4.0, now=120.0)
        self.assertTrue(dsf.dialogue_active(now=123.9))
        self.assertFalse(dsf.dialogue_active(now=124.1))
        self.assertIsNone(dsf.dialogue_source(now=124.1))

    def test_max_s_auto_expiry(self):
        dsf.begin_dialogue(_SRC, max_s=10.0, now=100.0)
        self.assertTrue(dsf.dialogue_active(now=109.0))
        self.assertFalse(dsf.dialogue_active(now=110.5))

    def test_stale_token_cannot_end_a_newer_dialogue(self):
        old = dsf.begin_dialogue(_SRC, now=100.0)
        new = dsf.begin_dialogue(_SRC, now=101.0)
        self.assertNotEqual(old, new)
        dsf.end_dialogue(old, tail_s=0.0, now=102.0)
        self.assertTrue(dsf.dialogue_active(now=103.0))
        dsf.end_dialogue(new, tail_s=0.0, now=104.0)
        self.assertFalse(dsf.dialogue_active(now=104.5))

    def test_second_end_does_not_extend_the_tail(self):
        tok = dsf.begin_dialogue(_SRC, now=100.0)
        dsf.end_dialogue(tok, tail_s=4.0, now=110.0)
        dsf.end_dialogue(tok, tail_s=4.0, now=113.0)
        self.assertFalse(dsf.dialogue_active(now=114.5))

    def test_entries_dropped_when_the_tail_ends(self):
        tok = dsf.begin_dialogue(_SRC, now=100.0)
        self.expect(window_s=20.0, now=100.0)      # would live to 124
        dsf.end_dialogue(tok, tail_s=4.0, now=105.0)
        self.assertIsNotNone(dsf.match(_LINE, now=108.0))
        self.assertFalse(dsf.dialogue_active(now=109.5))
        self.assertIsNone(dsf.match(_LINE, now=110.0))

    def test_other_sources_survive_the_tail(self):
        tok = dsf.begin_dialogue(_SRC, now=100.0)
        self.expect("A line from a different device entirely.",
                    window_s=20.0, now=100.0, source="other")
        dsf.end_dialogue(tok, tail_s=1.0, now=101.0)
        self.assertFalse(dsf.dialogue_active(now=103.0))
        self.assertIsNotNone(dsf.match(
            "a line from a different device entirely", now=103.0))


class ThreadSmokeTests(_Base):
    def test_concurrent_expect_and_match(self):
        errors = []

        def worker(k):
            try:
                for i in range(200):
                    h = dsf.expect(_SRC, f"Worker {k} says line number {i}.",
                                   window_s=1.0)
                    dsf.match(f"worker {k} says line number {i}")
                    dsf.expect_done(h)
                    dsf.dialogue_active()
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(k,))
                   for k in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertLessEqual(len(dsf._expected_snapshot()), dsf.EXPECT_MAX)


if __name__ == "__main__":
    unittest.main()
