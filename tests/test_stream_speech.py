"""Tests for core/stream_speech.py — the safe early-speech gate.

These pin the three safety contracts that make streamed TTS safe on JARVIS's
action-parse-BEFORE-speak pipeline:

  1. never release a sentence once '[' has appeared (action token starting),
  2. never release a sentence the preemptive-hallucination guard would cut,
  3. early-spoken sentences + remainder() == the final spoken_text (parity:
     no double-speak, no dropped words).

Pure stdlib unittest (no pytest / no audio / no LLM) — the speak sink is a list
and the hallucination detector is injected.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import stream_speech as ss  # noqa: E402


def _feed_chars(speaker, text):
    """Stream `text` one character at a time — the most adversarial chunking."""
    for ch in text:
        speaker.feed(ch)


def _feed_chunks(speaker, chunks):
    for c in chunks:
        speaker.feed(c)


class TestNormWs(unittest.TestCase):
    def test_collapses_and_strips(self):
        self.assertEqual(ss.norm_ws("  a   b\n\tc  "), "a b c")
        self.assertEqual(ss.norm_ws(""), "")
        self.assertEqual(ss.norm_ws(None), "")


class TestSplit(unittest.TestCase):
    def test_no_terminator_releases_nothing(self):
        sents, cur = ss.split_complete_sentences("hello there", 0)
        self.assertEqual(sents, [])
        self.assertEqual(cur, 0)

    def test_releases_confirmed_keeps_partial(self):
        sents, cur = ss.split_complete_sentences("Hi there. How ar", 0)
        self.assertEqual(sents, ["Hi there."])
        self.assertEqual("Hi there. How ar"[cur:], "How ar")

    def test_terminator_at_end_unconfirmed(self):
        # No char after the '?' yet — could be more streaming; don't release.
        sents, cur = ss.split_complete_sentences("How are you?", 0)
        self.assertEqual(sents, [])
        self.assertEqual(cur, 0)

    def test_decimal_not_split(self):
        sents, _ = ss.split_complete_sentences("It is 3.5 degrees out. Yes", 0)
        self.assertEqual(sents, ["It is 3.5 degrees out."])

    def test_multiple_sentences(self):
        sents, _ = ss.split_complete_sentences("A done. B done! C don", 0)
        self.assertEqual(sents, ["A done.", "B done!"])

    def test_closing_quote_after_terminator(self):
        sents, _ = ss.split_complete_sentences('He said "go." Then left', 0)
        self.assertEqual(sents, ['He said "go."'])

    def test_multiple_consecutive_terminators_consumed(self):
        # Repeated terminators ('!!') are absorbed into one boundary; the run
        # is consumed before the whitespace check.
        sents, _ = ss.split_complete_sentences("Wow!! Next thing", 0)
        self.assertEqual(sents, ["Wow!!"])


class TestEarlySpeakerHappyPath(unittest.TestCase):
    def test_multi_sentence_streams_prefix_leaves_last(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        full = "First part done. Second part also. Third trails off."
        _feed_chars(sp, full)
        # The last sentence is never early-spoken (terminator at end stays
        # buffered until the stream is known-complete → final path voices it).
        self.assertEqual(spoken, ["First part done.", "Second part also."])
        self.assertFalse(sp.aborted)
        rem = sp.remainder(full)
        self.assertEqual(rem, "Third trails off.")
        # Parity: what we spoke + remainder reproduces the full reply.
        self.assertEqual(ss.norm_ws(sp.spoken_concat() + " " + rem),
                         ss.norm_ws(full))

    def test_single_sentence_no_early_speak(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        full = "What time is it?"
        _feed_chars(sp, full)
        self.assertEqual(spoken, [])
        self.assertEqual(sp.remainder(full), full)

    def test_disabled_is_inert(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False, enabled=False)
        _feed_chars(sp, "A done. B done. C done.")
        self.assertEqual(spoken, [])
        self.assertFalse(sp.spoke_anything)
        self.assertEqual(sp.remainder("A done. B done. C done."),
                         "A done. B done. C done.")


class TestEarlySpeakerActionAbort(unittest.TestCase):
    def test_aborts_when_bracket_arrives_later(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        _feed_chunks(sp, ["Sure thing. ", "[ACTION:get_time]"])
        self.assertEqual(spoken, ["Sure thing."])
        self.assertTrue(sp.aborted)
        # parse_and_run_actions strips the token → spoken_text == "Sure thing."
        self.assertEqual(sp.remainder("Sure thing."), "")

    def test_bracket_same_chunk_speaks_nothing(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        sp.feed("Sure thing. [ACTION:get_time]")
        self.assertEqual(spoken, [])           # safe: never spoke the prefix
        self.assertTrue(sp.aborted)
        self.assertEqual(sp.remainder("Sure thing."), "Sure thing.")

    def test_action_only_reply_speaks_nothing(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        _feed_chars(sp, "[ACTION:get_time]")
        self.assertEqual(spoken, [])
        self.assertTrue(sp.aborted)


class TestEarlySpeakerHallucinationAbort(unittest.TestCase):
    def _detector(self, text):
        # Stand-in for the monolith's `_detect_preemptive_hallucination(t)
        # is not None` — flags execution claims.
        return ss.looks_like_execution_claim(text)

    def test_stops_at_hallucinated_claim(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, self._detector)
        _feed_chars(sp, "Of course. I'll set that reminder now. Done.")
        self.assertEqual(spoken, ["Of course."])
        self.assertTrue(sp.aborted)

    def test_refuse_case_remainder_is_benign(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, self._detector)
        _feed_chars(sp, "Of course. I'll set that reminder now.")
        self.assertEqual(spoken, ["Of course."])
        # If the guard REFUSES (spoken_text == "") the defensive branch keeps
        # us from re-speaking — remainder is the (empty) final text.
        self.assertEqual(sp.remainder(""), "")


class TestEarlySpeakerFailSafe(unittest.TestCase):
    def test_speak_raising_aborts(self):
        def boom(_):
            raise RuntimeError("tts down")
        sp = ss.EarlySpeaker(boom, lambda t: False)
        _feed_chars(sp, "First done. Second done. Third")
        # First sentence triggered the raise → abort, nothing further.
        self.assertTrue(sp.aborted)
        # remainder falls back to the full text (defensive non-prefix).
        self.assertEqual(sp.remainder("First done. Second done. Third"),
                         "First done. Second done. Third")

    def test_remainder_defensive_on_divergence(self):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        _feed_chars(sp, "Alpha here. Beta here. ")
        self.assertEqual(spoken, ["Alpha here.", "Beta here."])
        # Final text that does NOT start with what we spoke → speak full.
        diverged = "Completely different answer."
        self.assertEqual(sp.remainder(diverged), diverged)

    def test_empty_chunk_is_noop(self):
        # feed("") returns immediately without touching buffers or aborting.
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        sp.feed("")
        self.assertEqual(spoken, [])
        self.assertFalse(sp.aborted)
        self.assertFalse(sp.spoke_anything)

    def test_split_raising_aborts_feed(self):
        # A surprise error inside split_complete_sentences must abort early-speech
        # (fail-safe to the blocking path), not propagate out of feed().
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        with mock.patch.object(ss, "split_complete_sentences",
                               side_effect=RuntimeError("split boom")):
            sp.feed("A done. B done. ")    # must not raise
        self.assertTrue(sp.aborted)
        self.assertEqual(spoken, [])

    def test_remainder_with_only_whitespace_spoken_returns_full(self):
        # Defensive branch: _spoken is non-empty but normalises to '' (e.g. a
        # whitespace-only entry), so spoken_concat() is '' → remainder returns
        # the full final text rather than mis-subtracting.
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        sp._spoken.append("   ")            # white-box: force already == ""
        self.assertEqual(sp.spoken_concat(), "")
        self.assertEqual(sp.remainder("the full reply"), "the full reply")


class TestParityProperty(unittest.TestCase):
    """For action-free replies, early-spoken + remainder must == the reply,
    under every chunking — the invariant the live wiring relies on."""

    REPLIES = [
        "The weather looks clear today. Should be a fine afternoon, sir. Enjoy it.",
        "Right away. Pulling that up for you now. Here it comes.",
        "Yes. No. Maybe so. Final answer.",
        "It is 3.5 degrees and dropping. Bundle up. That is all.",
        "One sentence only and it trails",
        "Done.",
    ]

    def _check(self, reply, chunks):
        spoken = []
        sp = ss.EarlySpeaker(spoken.append, lambda t: False)
        _feed_chunks(sp, chunks)
        self.assertFalse(sp.aborted)
        rem = sp.remainder(reply)
        combined = ss.norm_ws(sp.spoken_concat() + " " + rem)
        self.assertEqual(combined, ss.norm_ws(reply),
                         msg=f"parity broke for {reply!r} chunks={chunks!r}")

    def test_char_by_char(self):
        for r in self.REPLIES:
            self._check(r, list(r))

    def test_whole_then_nothing(self):
        for r in self.REPLIES:
            self._check(r, [r])

    def test_two_way_splits(self):
        for r in self.REPLIES:
            for i in range(1, len(r)):
                self._check(r, [r[:i], r[i:]])


if __name__ == "__main__":
    unittest.main()
