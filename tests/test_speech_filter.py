"""Tests for core.speech_filter — the Whisper transcription gates extracted
from the monolith. These ran on every utterance with zero coverage before;
they pin the hallucination/confidence/length/always-accept gates and the
high-RMS confidence bypass."""
import unittest
from unittest import mock

import core.speech_filter as sf

NEUTRAL = {"no_speech_prob": 0.10, "avg_logprob": -0.30}
BAD = {"no_speech_prob": 0.95, "avg_logprob": -3.00}
# Whisper says "this IS speech" (low no_speech_prob) but is very unsure of the
# words (very negative avg_logprob) — isolates the avg_logprob gate.
LOW_LOGPROB = {"no_speech_prob": 0.10, "avg_logprob": -3.00}


class AmbientMusicTests(unittest.TestCase):
    def test_markers_detected(self):
        self.assertTrue(sf.is_ambient_music("[Music]"))
        self.assertTrue(sf.is_ambient_music("la la ♪ la"))
        self.assertTrue(sf.is_ambient_music("Music playing softly"))

    def test_plain_speech_not_music(self):
        self.assertFalse(sf.is_ambient_music("what time is it"))
        self.assertFalse(sf.is_ambient_music(""))


class ValidSpeechTests(unittest.TestCase):
    def test_empty_rejected(self):
        ok, reason = sf.is_valid_speech("", NEUTRAL)
        self.assertFalse(ok)
        self.assertEqual(reason, "empty")

    def test_always_accept_single_word(self):
        ok, _ = sf.is_valid_speech("yes", BAD)   # accepted before confidence gate
        self.assertTrue(ok)

    def test_hallucination_rejected(self):
        ok, reason = sf.is_valid_speech("thanks for watching", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("hallucination", reason)

    def test_too_short_chars(self):
        ok, reason = sf.is_valid_speech("a b", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("too short", reason)

    def test_valid_sentence(self):
        ok, _ = sf.is_valid_speech("what time is it", NEUTRAL)
        self.assertTrue(ok)

    def test_low_confidence_rejected(self):
        ok, reason = sf.is_valid_speech("some longer phrase here", BAD)
        self.assertFalse(ok)
        self.assertIn("no_speech_prob", reason)

    def test_high_rms_bypasses_confidence(self):
        # Loud, clearly-spoken audio is trusted even with bad Whisper scores.
        ok, _ = sf.is_valid_speech("turn it up", BAD, peak_rms=0.1)
        self.assertTrue(ok)

    def test_low_avg_logprob_rejected(self):
        # no_speech_prob passes its gate, but the avg_logprob confidence is
        # below WHISPER_MIN_AVG_LOGPROB → rejected with the logprob reason.
        ok, reason = sf.is_valid_speech("what time is it", LOW_LOGPROB)
        self.assertFalse(ok)
        self.assertIn("low confidence", reason)

    def test_single_long_word_too_few_words(self):
        # A single ≥4-char word that isn't an always-accept term clears the
        # char gate but trips the word-count gate (1 < WHISPER_MIN_WORDS) when
        # the audio wasn't loud enough to bypass it.
        ok, reason = sf.is_valid_speech("banana", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("1 words", reason)

    def test_missing_wake_word_rejected(self):
        # With a wake word configured, an utterance lacking it is dropped before
        # the confidence gates. Patched at module scope so the global is restored.
        with mock.patch.object(sf, "WAKE_WORD", "jarvis"):
            ok, reason = sf.is_valid_speech("what time is it", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("wake word", reason)

    def test_wake_word_present_passes(self):
        with mock.patch.object(sf, "WAKE_WORD", "jarvis"):
            ok, _ = sf.is_valid_speech("jarvis what time is it", NEUTRAL)
        self.assertTrue(ok)


class HallucinationVerdictTests(unittest.TestCase):
    """R10 (2026-09-29): a transcript that is ONLY a classic Whisper
    hallucination is judged on evidence, not on the word. Live 19:18:55: with
    nobody home, room noise at 1.49x the VAD threshold became "Bye." and was
    answered with a full LLM call, because the single-word always-accept
    shortcut ran before the hallucination check."""

    VAD = 0.008
    OK_CONF = {"no_speech_prob": 0.30, "avg_logprob": -0.60}

    def _v(self, text, peak=0.05, conf=None, **ctx):
        return sf.hallucination_verdict(
            text, self.OK_CONF if conf is None else conf, peak,
            vad_threshold=self.VAD, **ctx)

    def test_phrase_only_detection_ignores_case_and_punctuation(self):
        for t in ("Bye.", "bye", "Bye-bye!", "Thank you!", "You",
                  "Thanks for watching!", "Okay?", "[Music]"):
            self.assertTrue(sf.is_hallucination_only(t), t)
        for t in ("", "...", "thank you, sir", "bye for now",
                  "turn off the lights", "you know what"):
            self.assertFalse(sf.is_hallucination_only(t), t)

    def test_the_live_incident_is_noise(self):
        # Fresh session (the owner never spoke), JARVIS's greeting asked a
        # question 8 s earlier, peak 0.0119 on a 0.008 threshold.
        verdict, why = self._v("Bye.", peak=0.0119, owner_idle_s=None,
                               since_jarvis_s=8.0, jarvis_asked=True)
        self.assertEqual(verdict, "noise")
        self.assertIn("1.49x", why)
        # Without the verdict, is_valid_speech ACCEPTS it — the bug.
        self.assertTrue(sf.is_valid_speech("Bye.", self.OK_CONF, 0.0119)[0])

    def test_owner_idle_and_no_question_is_noise_even_when_loud(self):
        verdict, why = self._v("Thank you.", peak=0.05, owner_idle_s=900.0,
                               since_jarvis_s=600.0)
        self.assertEqual(verdict, "noise")
        self.assertIn("owner silent 900 s", why)
        self.assertEqual(self._v("Bye.", peak=0.05)[0], "noise",
                         "no owner turn this session at all")

    def test_poor_whisper_confidence_is_noise_even_in_a_reply(self):
        for conf in ({"no_speech_prob": 0.95, "avg_logprob": -0.4},
                     {"no_speech_prob": 0.10, "avg_logprob": -2.5}):
            verdict, why = self._v("Thank you.", peak=0.2, conf=conf,
                                   owner_idle_s=5.0, since_jarvis_s=2.0)
            self.assertEqual(verdict, "noise", conf)
            self.assertIn("whisper", why)

    def test_thank_you_right_after_jarvis_answered_is_a_reply(self):
        # The owner's quiet mic: 1.2x the threshold would be "marginal", but
        # the conversation makes it clearly a reply.
        verdict, _ = self._v("Thank you.", peak=0.0096, owner_idle_s=12.0,
                             since_jarvis_s=2.5)
        self.assertEqual(verdict, "reply")
        self.assertTrue(sf.is_valid_speech("Thank you.", self.OK_CONF,
                                           0.0096, reply=True)[0])
        self.assertFalse(sf.is_valid_speech("Thank you.", self.OK_CONF,
                                            0.0096)[0],
                         "without the verdict it stays a hallucination match")

    def test_bye_ending_a_conversation_is_a_reply(self):
        for text in ("Bye.", "Bye bye!", "Okay.", "Yeah.", "Thanks."):
            verdict, _ = self._v(text, peak=0.0090, owner_idle_s=40.0,
                                 since_jarvis_s=4.0)
            self.assertEqual(verdict, "reply", text)
            self.assertTrue(sf.is_valid_speech(text, self.OK_CONF, 0.009,
                                               reply=True)[0], text)

    def test_a_pending_prompt_counts_only_while_the_owner_is_around(self):
        self.assertEqual(self._v("Yeah.", peak=0.009, owner_idle_s=60.0,
                                 since_jarvis_s=45.0,
                                 prompt_pending=True)[0], "reply")
        # A stale confirmation must not be answered by room noise.
        self.assertEqual(self._v("Okay.", peak=0.009, owner_idle_s=3600.0,
                                 since_jarvis_s=3000.0,
                                 prompt_pending=True)[0], "noise")

    def test_non_reply_phrases_are_never_a_reply(self):
        for text in ("You", "Thanks for watching!", "Please subscribe", "Hmm"):
            verdict, _ = self._v(text, peak=0.009, owner_idle_s=5.0,
                                 since_jarvis_s=1.0)
            self.assertEqual(verdict, "noise", text)   # marginal level
            verdict, _ = self._v(text, peak=0.05, owner_idle_s=5.0,
                                 since_jarvis_s=1.0)
            self.assertEqual(verdict, "", text)        # is_valid_speech decides
            self.assertFalse(sf.is_valid_speech(text, self.OK_CONF, 0.05,
                                                reply=True)[0], text)

    def test_answer_to_a_question_with_an_idle_owner_falls_through(self):
        # Clear audio answering a question JARVIS just asked: not noise; the
        # old is_valid_speech rule then decides ("bye" accepted).
        verdict, _ = self._v("Bye.", peak=0.03, owner_idle_s=None,
                             since_jarvis_s=3.0, jarvis_asked=True)
        self.assertEqual(verdict, "")

    def test_ordinary_speech_is_never_touched(self):
        for text in ("turn off the lights", "what time is it", "thank you sir"):
            self.assertEqual(self._v(text, peak=0.0081, owner_idle_s=None),
                             ("", ""), text)

    def test_reason_never_contains_the_transcript(self):
        for kw in ({"peak": 0.0085}, {"owner_idle_s": 999.0},
                   {"conf": {"no_speech_prob": 0.99, "avg_logprob": -3}}):
            verdict, why = self._v("Thanks for watching!", **kw)
            self.assertEqual(verdict, "noise")
            self.assertNotIn("thank", why.lower())
            self.assertNotIn("watching", why.lower())

    def test_never_raises(self):
        self.assertEqual(sf.hallucination_verdict("Bye.", None, None,
                                                  vad_threshold="x"), ("", ""))
        self.assertEqual(sf.hallucination_verdict(None, {}, 0.0,
                                                  vad_threshold=0.008),
                         ("", ""))

    def test_knobs_are_overridable_and_consumed(self):
        self.addCleanup(sf.reset_overrides)
        ctx = {"owner_idle_s": 30.0, "since_jarvis_s": 60.0}
        self.assertEqual(self._v("Bye.", peak=0.013, **ctx)[0], "")
        self.assertEqual(sf.apply_overrides({"NOISE_RMS_MARGIN": 2.0}),
                         {"NOISE_RMS_MARGIN": 2.0})
        self.assertEqual(self._v("Bye.", peak=0.013, **ctx)[0], "noise")
        sf.apply_overrides({"NOISE_REPLY_WINDOW_S": 90.0})
        self.assertEqual(self._v("Bye.", peak=0.013, **ctx)[0], "reply")
        sf.apply_overrides({"NOISE_OWNER_IDLE_S": 10.0})
        self.assertEqual(self._v("Bye.", peak=0.05, **ctx)[0], "noise")
        self.assertEqual(sf.apply_overrides({"NOISE_RMS_MARGIN": 0.5}), {},
                         "a margin under 1x would drop nothing")


class RepetitionNoiseTests(unittest.TestCase):
    """2026-09-30: a degenerate transcript is noise in ANY context. Live
    session_2026-09-29_22-06-02.log 22:53:08: mid-conversation, music in the
    room, Whisper turned 9.8 s of audio into "I I I I I I I I I I I I I" and
    JARVIS answered it ("Very good, sir.") — it is not a known hallucination
    phrase and is_valid_speech saw thirteen words."""

    LIVE = "I I I I I I I I I I I I I"
    VAD = 0.008
    OK_CONF = {"no_speech_prob": 0.30, "avg_logprob": -0.60}
    # The owner in a conversation, JARVIS answered 5 s ago: the context in
    # which the phrase gate would have KEPT a reply.
    TALKING = {"owner_idle_s": 20.0, "since_jarvis_s": 5.0}

    def _v(self, text, peak=0.0119, **ctx):
        return sf.hallucination_verdict(text, self.OK_CONF, peak,
                                        vad_threshold=self.VAD, **ctx)

    def test_the_live_transcript_is_noise_mid_conversation(self):
        verdict, why = self._v(self.LIVE, **self.TALKING)
        self.assertEqual((verdict, why), ("noise", "1 distinct word in 13"))
        # Loud and confident does not rescue it either.
        self.assertEqual(self._v(self.LIVE, peak=0.2, **self.TALKING)[0],
                         "noise")
        # Without the gate is_valid_speech ACCEPTS it — the bug.
        self.assertTrue(sf.is_valid_speech(self.LIVE, self.OK_CONF, 0.0119)[0])

    def test_degenerate_shapes_are_noise(self):
        cases = {
            "you you you": "1 distinct word in 3",
            "The the the the.": "1 distinct word in 4",
            "Thank you. Thank you. Thank you. Thank you. Thank you.":
                "2 words are 100% of 10",
            "I I I I I you I I I I": "2 words are 100% of 10",
            "uh um uh": "filler run of 3 words",
            "Uh, um, er, um.": "filler run of 4 words",
            "la la la la la": "1 distinct word in 5",
            "go to go to go to go to go to go": "2 words are 100% of 11",
            "so it was so it was so it was so it was":
                "lexical diversity 0.25 over 12 words",
        }
        for text, why in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self._v(text, **self.TALKING),
                                 ("noise", why))

    def test_stop_and_confirmation_words_are_exempt(self):
        for text in ("stop", "stop stop stop", "Stop! Stop! Stop! Stop!",
                     "yes yes", "yes yes yes", "no no no",
                     "No, no, no, no, no.", "wait wait wait",
                     "okay okay okay", "Jarvis, Jarvis, Jarvis",
                     "hello hello hello", "testing testing testing"):
            with self.subTest(text=text):
                self.assertEqual(sf.repetition_reason(text), "")
                self.assertNotEqual(self._v(text, peak=0.05,
                                            **self.TALKING)[0], "noise")

    def test_real_speech_with_repeats_passes(self):
        for text in ("oh no no no", "I I I think so", "no I don't think so",
                     "come on come on", "turn it up, up, up",
                     "what time is it", "turn off the lights",
                     "I said I wanted the other one, not that one"):
            with self.subTest(text=text):
                self.assertEqual(sf.repetition_reason(text), "")

    def test_too_short_to_judge_is_left_to_the_other_gates(self):
        for text in ("I I", "uh um", "", None):
            with self.subTest(text=text):
                self.assertEqual(sf.repetition_reason(text), "")

    def test_reason_is_numbers_only(self):
        for text in (self.LIVE, "banana banana banana banana",
                     "uh um uh um", "red red red red blue red red red red"):
            with self.subTest(text=text):
                why = sf.repetition_reason(text)
                self.assertTrue(why)
                for word in set(sf._norm_phrase(text).split()):
                    self.assertNotRegex(why.lower(), r"\b" + word + r"\b")

    def test_never_raises(self):
        self.assertEqual(sf.repetition_reason(object()), "")
        self.assertEqual(sf.repetition_reason(12345), "")


if __name__ == "__main__":
    unittest.main()
