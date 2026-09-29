"""Tests for core/device_speech_filter.py — known-device speech must never
command JARVIS.

GENERIC fixtures only: a made-up "desk speaker" device with made-up lines. The
real phrase lists are private and live only in the gitignored
data/device_phrases/ directory on the box.

stdlib unittest only; CI-safe (no audio, no monolith).
    python tools/run_tests.py test_device_speech_filter
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from core import device_speech_filter as dsf

_FIXTURE = {
    "source": "desk speaker",
    "phrases": [
        "Desk speaker ready to play.",
        "Battery level is getting low.",
        "Please stop poking my buttons.",
        "Volume up.",
        "Hello!",
        "Tray empty.",
        "Lamp warming up.",
        "Jarvis, desk lamp warmed.",
        "Tray not ready.",
        "Calibration sequence is done.",
        "Battery drained, motors stopping now.",
        # Generic owner vocabulary a device may also say (R3 review).
        "Yes.",
        "Next track",
        "Maybe.",
        "Got it.",
        "Go to sleep.",
    ],
}
_WAKE = {"jarvis", "hey jarvis"}


class _DirCase(unittest.TestCase):
    def setUp(self):
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        self.tmp = tempfile.mkdtemp(prefix="dsf_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dir = os.path.join(self.tmp, "device_phrases")
        os.makedirs(self.dir)

    def write(self, name, payload, raw=False):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload if raw else json.dumps(payload))
        return path

    def match(self, text, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return dsf.match(text, directory=self.dir, **kw)


class NormaliseTests(unittest.TestCase):
    def test_lowercase_punctuation_whitespace(self):
        self.assertEqual(dsf.normalise("  Desk   Speaker, READY to play!! "),
                         "desk speaker ready to play")

    def test_apostrophes_dropped_hyphens_split(self):
        self.assertEqual(dsf.normalise("It's an e-stop"), "its an e stop")

    def test_non_string(self):
        self.assertEqual(dsf.normalise(None), "")
        self.assertEqual(dsf.normalise(42), "")


class MatchTests(_DirCase):
    def setUp(self):
        super().setUp()
        self.write("desk.json", _FIXTURE)

    # ── filtered ──────────────────────────────────────────────────────────
    def test_exact_phrase_is_filtered_with_its_source(self):
        hit = self.match("Desk speaker ready to play.")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "desk speaker")
        self.assertEqual(hit[2], 1.0)

    def test_misheard_variant_at_or_above_080_is_filtered(self):
        # "desk" misheard as "the": ratio 0.902 against the phrase.
        hit = self.match("The speaker ready to play.")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "desk speaker")
        self.assertGreaterEqual(hit[2], 0.80)
        self.assertIsNotNone(self.match("battery level is getting slow"))

    def test_truncated_prefix_meeting_the_ratio_is_filtered(self):
        # Whisper cut the line short but kept most of it (ratio 0.923).
        self.assertIsNotNone(self.match("Battery level is getting"))

    def test_two_word_exact_phrase_is_filtered(self):
        self.assertIsNotNone(self.match("Tray, EMPTY!"))

    def test_whisper_noise_around_a_line_is_filtered(self):
        # Leading filler / trailing hallucinated "you" / "thank you".
        for text in ("Oh, desk speaker ready to play.",
                     "Desk speaker ready to play you",
                     "Desk speaker ready to play. Thank you."):
            self.assertIsNotNone(self.match(text), text)

    def test_stopping_is_not_a_stop_command(self):
        # Exact stop words only: a device line saying "stopping" is filtered.
        self.assertIsNotNone(self.match("Battery drained, motors stopping now."))
        self.assertFalse(dsf.has_stop_word("motors stopping"))
        self.assertTrue(dsf.has_stop_word("stop the motors"))

    def test_wake_led_short_line_misheard_is_filtered(self):
        self.assertIsNotNone(self.match("Jarvis desk lamp warned",
                                        wake_phrases=_WAKE))

    # ── passes ────────────────────────────────────────────────────────────
    def test_stop_word_is_never_filtered_even_when_similar(self):
        # Verbatim device line that contains a stop word...
        self.assertIsNone(self.match("Please stop poking my buttons."))
        # ...and owner stop commands close to other device lines.
        for text in ("stop, desk speaker ready to play",
                     "Halt battery level is getting low",
                     "e-stop desk speaker ready to play",
                     "estop", "abort", "freeze", "emergency",
                     "cancel volume up", "e-stop", "stop"):
            self.assertIsNone(self.match(text), text)
        self.assertTrue(dsf.has_stop_word("E-Stop now"))
        self.assertFalse(dsf.has_stop_word("desk speaker ready to play"))

    def test_unrelated_command_below_080_passes(self):
        self.assertIsNone(self.match("what's the battery level on my laptop"))
        self.assertIsNone(self.match("open my bookmarks"))
        # Within the word-count guard, so only the RATIO keeps these out:
        # 0.772 and 0.571 against "battery level is getting low".
        self.assertIsNone(self.match("the battery level is very low"))
        self.assertIsNone(self.match("what level is the battery at"))

    def test_threshold_boundary_is_inclusive_at_080(self):
        # Exactly 0.80 against the long "battery level is getting low".
        hit = self.match("battery low is getting")
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[2], 0.80, places=6)
        self.assertEqual(dsf.FUZZY_MIN_RATIO, 0.80)

    def test_owner_confirmation_passes_even_when_a_device_says_it(self):
        # The owner's bare "yes" to a pending confirmation, media transport
        # and common replies are never swallowed by a device list holding
        # the same generic line (R3 review: the gate runs before
        # handle_confirmation_response).
        for text in ("Yes.", "yes", "next track", "Next track.", "maybe",
                     "got it", "Hello!", "volume up"):
            self.assertIsNone(self.match(text), text)
        # ...and a caller-protected phrase (the monolith passes its sleep /
        # shutdown-prompt phrases) likewise.
        self.assertIsNotNone(self.match("go to sleep"))
        self.assertIsNone(self.match("Go to sleep.", never_match={"go to sleep"}))

    def test_short_owner_utterance_never_fuzzy_matches(self):
        # A 1-2 word UTTERANCE is exact-only too, whatever the phrase length:
        # 0.833 against the long "calibration sequence is done", and 0.889
        # against the 3-word "lamp warming up".
        self.assertIsNone(self.match("calibration sequence"))
        self.assertIsNone(self.match("lamp warming"))
        self.assertIsNotNone(self.match("calibration sequence is don"))

    def test_short_phrase_needs_the_stricter_ratio(self):
        # "tray is ready" vs the 14-char "tray not ready": 0.815 — one
        # different word on a short line, the owner, not the device.
        self.assertIsNone(self.match("tray is ready"))
        self.assertEqual(dsf.SHORT_FUZZY_MIN_RATIO, 0.90)

    def test_shared_wake_prefix_does_not_inflate_the_score(self):
        # 0.87 whole-vs-whole against the long "jarvis desk lamp warmed"
        # (>= 0.80), but once the shared "jarvis" is dropped the remainder is
        # a short line scoring 0.812 (< 0.90): one different word, the owner.
        self.assertIsNone(self.match("Jarvis, desk lamp dimmed.",
                                     wake_phrases=_WAKE))
        self.assertIsNotNone(self.match("Jarvis, desk lamp dimmed."))

    def test_two_word_near_miss_passes(self):
        # ~0.95 fuzzily, but 1-2 word phrases need an exact match.
        self.assertIsNone(self.match("tray empy"))
        self.assertIsNone(self.match("tray empty please"))

    def test_phrase_inside_a_longer_owner_sentence_passes(self):
        self.assertIsNone(self.match(
            "tell me why desk speaker ready to play keeps printing"))
        # Ratio 0.929 whole-vs-whole, but it is the phrase verbatim + a word.
        self.assertIsNone(self.match("desk speaker ready to play now"))

    def test_owner_command_built_around_a_line_passes(self):
        # Ratio 0.867, but two words longer than the 5-word phrase.
        self.assertIsNone(self.match("desk speaker are you ready to play"))

    def test_protected_wake_phrase_is_never_filtered(self):
        self.assertIsNotNone(self.match("tray empty"))
        self.assertIsNone(self.match("Tray empty.", never_match={"tray empty"}))

    def test_empty_utterance(self):
        self.assertIsNone(self.match(""))
        self.assertIsNone(self.match("   ?! "))
        self.assertIsNone(self.match(None))


class LoadingTests(_DirCase):
    def test_missing_dir_means_no_filtering(self):
        missing = os.path.join(self.tmp, "nope")
        self.assertEqual(dsf.load_phrases(missing), [])
        self.assertIsNone(dsf.match("Desk speaker ready to play.",
                                    directory=missing))

    def test_invalid_files_mean_no_filtering_and_no_exception(self):
        self.write("broken.json", "{not json", raw=True)
        self.write("list.json", ["Desk speaker ready to play."])
        self.write("nophrases.json", {"source": "x", "phrases": "oops"})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertIsNone(dsf.match("Desk speaker ready to play.",
                                        directory=self.dir))
        out = buf.getvalue()
        # Reported by file NAME only, never contents.
        self.assertIn("broken.json", out)
        self.assertNotIn("ready to play", out.lower())

    def test_a_bad_file_does_not_disable_a_good_one(self):
        self.write("broken.json", "{", raw=True)
        self.write("desk.json", _FIXTURE)
        self.assertIsNotNone(self.match("Desk speaker ready to play."))

    def test_source_falls_back_to_the_file_stem(self):
        self.write("lamp.json", {"phrases": ["The lamp is warming up."]})
        hit = self.match("the lamp is warming up")
        self.assertEqual(hit[0], "lamp")

    def test_cache_reloads_when_a_file_changes(self):
        path = self.write("desk.json", {"source": "a", "phrases": ["one two three"]})
        self.assertEqual(self.match("one two three")[0], "a")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"source": "bb", "phrases": ["four five six seven"]}, f)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        self.assertIsNone(self.match("one two three"))
        self.assertEqual(self.match("four five six seven")[0], "bb")

    def test_cache_does_not_reparse_an_unchanged_dir(self):
        self.write("desk.json", _FIXTURE)
        self.match("tray empty")
        with mock.patch.object(dsf, "_parse_file",
                               side_effect=AssertionError("re-parsed")):
            self.assertIsNotNone(self.match("tray empty"))

    def test_internal_error_fails_open(self):
        self.write("desk.json", _FIXTURE)
        with mock.patch.object(dsf, "load_phrases",
                               side_effect=RuntimeError("boom")):
            self.assertIsNone(dsf.match("tray empty", directory=self.dir))

    def test_default_dir_resolves_through_core_paths(self):
        # JARVIS_DATA_DIR (the test/staging redirect) wins over the live data/.
        self.write("desk.json", _FIXTURE)
        with mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.tmp}):
            self.assertEqual(dsf.phrases_dir(), self.dir)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNotNone(dsf.match("Desk speaker ready to play."))


class GitignoreTests(unittest.TestCase):
    def test_phrase_dir_is_gitignored(self):
        # The phrase lists are private: data/device_phrases/ must be covered
        # by the repo's data/* ignore rule (no negation re-including it).
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".gitignore"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        self.assertIn("data/*", lines)
        self.assertFalse([ln for ln in lines
                          if ln.startswith("!") and "device_phrases" in ln])


if __name__ == "__main__":
    unittest.main()
