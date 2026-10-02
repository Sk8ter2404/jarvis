"""core/wake_prefix.py: the ONE wake-prefix rule (2026-10-01).

Live 2026-10-01 20:57, wake-word mode on: "What Jarvis what model are you?"
was dropped ("[bg-audio] wake-word mode — ignoring non-wake utterance")
because the gate only took "Jarvis" as the FIRST word. The wake word may now
be word 1, 2 or 3 behind lead interjections; a mid-sentence mention is still
refused.

Stdlib unittest, CI-safe: the monolith is never imported
(tests/monolith/test_monolith_wake_prefix.py covers the wiring).

    python -m unittest tests.test_wake_prefix
"""
from __future__ import annotations

import os
import re
import unittest

from core import wake_prefix as wp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Addressed to JARVIS: the legacy forms, and the wake word behind 1-2 lead
# interjections.
ACCEPTED = (
    "Jarvis",
    "Jarvis.",
    "Jarvis, what time is it?",
    "JARVIS turn off the lights",
    "hey Jarvis, play some jazz",
    "Hey, Jarvis, play some jazz",
    "ok jarvis",
    "Okay Jarvis, go ahead",
    "What Jarvis what model are you?",          # the live line
    "What? Jarvis, what's the weather?",
    "Um, Jarvis, pause the music",
    "uh Jarvis",
    "Oh Jarvis, turn it down",
    "Yo Jarvis what's up",
    "So, Jarvis, what's next on my calendar",
    "Alright, Jarvis.",
    "alright jarvis lights off",
    "Okay, so Jarvis, lights off",
    "um okay Jarvis set a timer",
    "All right, Jarvis, lights off",           # Whisper's "alright"
    "hey hey Jarvis",
    "What - Jarvis - what time is it",        # a bare dash is not a word
    "Jarvis said",                            # legacy form: unchanged
    "hey Jarvis says hi",                     # legacy form: unchanged
)

# Not addressed to JARVIS.
REFUSED = (
    "",
    "   ",
    "what time is it",
    "I asked Jarvis yesterday about the weather",
    "tell Jarvis that dinner is ready",
    "and Jarvis too",
    "so um uh Jarvis play music",              # wake word is word 4
    "okay okay okay Jarvis",                   # word 4
    "um all right Jarvis lights off",          # word 4 ("all right" = 2)
    "hey what is Jarvis doing",
    "the Jarvis thing is broken",
    "Jarvis's voice is weird",                 # a possessive, not the name
    "what's Jarvis up to",                     # "what's" is not a filler
    "So Jarvis said it would rain",            # filler-led mention
    "What Jarvis told me was wrong",
    "oh Jarvis thinks he's funny",
    "um Jarvis keeps cutting me off",
)


class HasWakePrefixTests(unittest.TestCase):
    def test_accepted(self):
        for text in ACCEPTED:
            with self.subTest(text=text):
                self.assertTrue(wp.has_wake_prefix(text))

    def test_refused(self):
        for text in REFUSED:
            with self.subTest(text=text):
                self.assertFalse(wp.has_wake_prefix(text))

    def test_non_strings_are_never_a_wake(self):
        for bad in (None, 42, b"jarvis", ["jarvis"]):
            with self.subTest(bad=bad):
                self.assertFalse(wp.has_wake_prefix(bad))

    def test_position_is_capped_at_word_three(self):
        self.assertEqual(wp.MAX_WAKE_POSITION, 3)
        self.assertTrue(wp.has_wake_prefix("um uh jarvis go"))
        self.assertFalse(wp.has_wake_prefix("um uh um jarvis go"))
        self.assertTrue(wp.has_wake_prefix("all right jarvis go"))
        self.assertFalse(wp.has_wake_prefix("so all right jarvis go"))
        # A long utterance is still judged by its first words only.
        self.assertTrue(wp.has_wake_prefix("um Jarvis " + "word " * 500))
        self.assertFalse(wp.has_wake_prefix("word " * 500 + "Jarvis"))

    def test_the_spec_fillers_are_all_accepted(self):
        for f in ("what", "okay", "hey", "yo", "so", "um", "uh", "oh",
                  "alright", "ok"):
            with self.subTest(filler=f):
                self.assertIn(f, wp.WAKE_LEAD_FILLERS)
                self.assertTrue(wp.has_wake_prefix(f"{f} jarvis play jazz"))

    def test_no_content_word_is_a_filler(self):
        # A content word would turn a mention into a wake.
        for w in ("i", "tell", "ask", "asked", "and", "the", "please",
                  "can", "is", "did"):
            with self.subTest(word=w):
                self.assertNotIn(w, wp.WAKE_LEAD_FILLERS)


class StripWakeLeadTests(unittest.TestCase):
    CASES = (
        ("What Jarvis what model are you?", "what model are you?"),
        ("Um, Jarvis, pause the music", "pause the music"),
        ("Jarvis, what time is it?", "what time is it?"),
        ("hey Jarvis, play jazz", "play jazz"),
        ("Hey, Jarvis - play jazz", "play jazz"),
        ("Okay, so Jarvis: lights off", "lights off"),
        ("Jarvis", ""),
        ("Oh, Jarvis.", ""),
    )

    def test_filler_and_wake_word_are_stripped(self):
        for text, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(wp.strip_wake_lead(text), want)

    def test_text_not_led_by_the_wake_word_is_unchanged(self):
        for text in ("I asked Jarvis yesterday", "what time is it",
                     "So Jarvis said it would rain", ""):
            with self.subTest(text=text):
                self.assertEqual(wp.strip_wake_lead(text), text)
        self.assertEqual(wp.strip_wake_lead(None), "")


class CanonicalWakeTextTests(unittest.TestCase):
    def test_a_filler_led_command_gets_the_plain_prefix_form(self):
        for text, want in (
                ("What Jarvis what model are you?",
                 "Jarvis what model are you?"),
                ("Um, Jarvis, pause the music", "Jarvis, pause the music"),
                ("Okay, so Jarvis, lights off", "Jarvis, lights off"),
                ("All right, Jarvis, lights off", "Jarvis, lights off"),
                ("hey hey Jarvis play jazz", "Jarvis play jazz")):
            with self.subTest(text=text):
                got = wp.canonical_wake_text(text)
                self.assertEqual(got, want)
                # ...which the legacy first-word rule already accepted.
                self.assertTrue(got.lower().startswith("jarvis"))

    def test_legacy_forms_are_never_rewritten(self):
        for text in ("Jarvis, play jazz", "hey Jarvis, play jazz",
                     "Okay Jarvis, go ahead", "ok jarvis yes"):
            with self.subTest(text=text):
                self.assertEqual(wp.canonical_wake_text(text), text)

    def test_a_name_that_ends_the_utterance_keeps_its_lead(self):
        # "Alright, Jarvis." answers a question: the lead IS the message.
        for text in ("Alright, Jarvis.", "All right, Jarvis.", "So Jarvis",
                     "um okay Jarvis", "What, Jarvis?"):
            with self.subTest(text=text):
                self.assertEqual(wp.canonical_wake_text(text), text)

    def test_unaddressed_text_is_unchanged(self):
        for text in REFUSED:
            with self.subTest(text=text):
                self.assertEqual(wp.canonical_wake_text(text), text)


class NoStaleDuplicateTests(unittest.TestCase):
    """Every copy of the leading-wake rule now calls core.wake_prefix."""

    def _src(self, *parts):
        with open(os.path.join(_ROOT, *parts), encoding="utf-8") as f:
            return f.read()

    def test_fast_paths_recall_cleaner_uses_the_helper(self):
        from core import fast_paths as fp
        self.assertFalse(hasattr(fp, "_WAKE_LEAD_RE"),
                         "the stale wake-lead regex is back")
        hist = [{"role": "user",
                 "content": "Um, Jarvis, what's the capital of France?"},
                {"role": "assistant", "content": "Paris, sir."}]
        self.assertEqual(fp.prior_owner_utterance(hist),
                         "what's the capital of France")

    def test_standby_music_gate_uses_the_helper(self):
        src = self._src("skills", "standby_audio_detect.py")
        body = src[src.index("def should_refuse_wake("):]
        body = body[:body.index("\ndef ", 10)]
        self.assertIn("has_wake_prefix(", body)
        self.assertNotIn('"jarvis"', body)

    def test_monolith_gate_uses_the_helper(self):
        src = self._src("bobert_companion.py")
        body = src[src.index("def _text_has_wake_prefix("):]
        body = body[:body.index("\ndef ", 10)]
        self.assertIn("has_wake_prefix(", body)
        self.assertNotIn('"jarvis"', body)
        carry = src[src.index("def _standby_wake_carries_command("):]
        carry = carry[:carry.index("\ndef ", 10)]
        self.assertIn("strip_wake_lead(", carry)
        self.assertNotIn("_WAKE_LEAD_RE", src)

    def test_the_dashboard_standby_guard_mirrors_the_rule(self):
        # The page decides in the browser whether a typed command in standby
        # needs a wake first; its regex must take the same fillers, at most
        # MAX_WAKE_POSITION - 1 of them.
        src = self._src("tools", "web_interface.py")
        m = re.search(r"const WAKE_WORD_RE = /(.+)/i;", src)
        self.assertIsNotNone(m, "WAKE_WORD_RE literal not found")
        js = m.group(1)
        alt = re.search(r"\(\?:\(\?:([^()]+)\)", js)
        self.assertIsNotNone(alt, js)
        page = {a.replace(r"\s+", " ") for a in alt.group(1).split("|")}
        self.assertEqual(page, set(wp.WAKE_LEAD_FILLERS)
                         | {" ".join(p) for p in wp.WAKE_LEAD_PHRASES})
        self.assertIn("{0,%d}" % (wp.MAX_WAKE_POSITION - 1), js)
        rx = re.compile(js, re.IGNORECASE)
        for text in ACCEPTED:
            with self.subTest(accepted=text):
                self.assertIsNotNone(rx.search(text))
        # The mention guard (and the possessive) are server-side only: the
        # page then just skips its "wake him first?" prompt and the standby
        # handler still refuses to run the line as a command.
        server_only = {"So Jarvis said it would rain",
                       "What Jarvis told me was wrong",
                       "oh Jarvis thinks he's funny",
                       "um Jarvis keeps cutting me off",
                       "Jarvis's voice is weird",
                       # the page counts fillers, not words
                       "um all right Jarvis lights off"}
        for text in REFUSED:
            if text in server_only:
                continue
            with self.subTest(refused=text):
                self.assertIsNone(rx.search(text))


if __name__ == "__main__":
    unittest.main()
