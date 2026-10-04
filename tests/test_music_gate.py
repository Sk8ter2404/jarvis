"""core/music_gate.py — stop transcribing music, keep hearing the owner
(2026-10-04). Pure policy + the per-minute counter; stdlib only.

THE LIVE NUMBERS (2026-10-04 13:49-15:40 session, wake-word mode, music on
the speakers): every capture ran to the 30 s cap, Parakeet decoded it, Whisper
re-decoded it as a "no-wake" rescue (427 rescues of 536 Parakeet decodes; 1,381
rescues 10-02..10-04 made 3 owner turns), and the ambient listener ran Whisper
on every 2.5 s of the same frames (~15 a minute): ~35 % of the 1650 and ~39
CPU-s a minute on lyrics.

Pinned here:
  * music mode = audio playing AND only a wake-word line gets through;
  * the wake hint ("Jarvis"-like word in the first three) and that the
    voice is asked only in 'on' and only when the text gives no hint —
    'shadow' asks it only AFTER a rescue that made a wake line
    (shadow_lost; review 2026-10-04: asking before the rescue cost the main
    loop 18-64 ms a music capture for a rescue that runs anyway);
  * the rescue / ambient decisions for off, shadow and on — and that the
    gate has NO capture cut (review 2026-10-04: nothing is recorded between
    captures, so a 10 s cut tripled the deaf gaps over music);
  * one counter line per minute WITH music, numbers only.

    python -m unittest tests.test_music_gate
"""
from __future__ import annotations

import unittest

from core import music_gate as mg


class ModeTests(unittest.TestCase):
    def test_modes(self):
        for v, want in (("off", "off"), ("SHADOW", "shadow"), (" on ", "on"),
                        ("", "off"), ("yes", "off"), (None, "off"),
                        (True, "off")):
            self.assertEqual(mg.mode_setting(v), want)
        self.assertEqual(mg.DEFAULT_MODE, "shadow")

    def test_music_mode_needs_audio_and_a_wake_word_rule(self):
        self.assertFalse(mg.music_mode(playing=False, wake_mode=True))
        self.assertFalse(mg.music_mode(playing=True))   # talks normally
        for kw in ("standby", "wake_mode", "music_refuse"):
            self.assertTrue(mg.music_mode(playing=True, **{kw: True}), kw)


class WakeHintTests(unittest.TestCase):
    def test_sounds_like_jarvis_in_the_first_three_words(self):
        for text in ("Jarvis, pause the music", "Travis pause", "Jervis",
                     "hey javis what's new", "Um, Garvis, next song",
                     "oh jarves"):
            self.assertTrue(mg.wake_hint(text), text)

    def test_lyrics_give_no_hint(self):
        for text in ("", None, "la la la love", "the service is down",
                     "I want it that way, Jarvis",          # word 5
                     "in the jar", "drivers on the road"):
            self.assertFalse(mg.wake_hint(text), text)


class RescueTests(unittest.TestCase):
    def setUp(self):
        self.asked = 0

    def _voice(self, verdict):
        def fn():
            self.asked += 1
            return verdict
        return fn

    def test_off_or_no_music_always_rescues(self):
        for mode, music in (("off", True), ("on", False), ("shadow", False)):
            self.assertEqual(mg.rescue_decision(
                mode, music, "no-wake", "la la", self._voice(mg.NOT_OWNER)),
                "")
        self.assertEqual(self.asked, 0)

    def test_only_empty_and_no_wake_are_gated(self):
        self.assertEqual(mg.rescue_decision(
            "on", True, "check-failed", "", self._voice(mg.NOT_OWNER)), "")
        self.assertEqual(self.asked, 0)

    def test_on(self):
        v = self._voice(mg.NOT_OWNER)
        self.assertEqual(mg.rescue_decision("on", True, "no-wake",
                                            "la la love", v), "skip")
        self.assertEqual(mg.rescue_decision("on", True, "empty", "", v),
                         "skip")
        for who in (mg.OWNER, mg.UNSURE, mg.UNAVAILABLE):  # fail open
            self.assertEqual(mg.rescue_decision(
                "on", True, "no-wake", "la la", self._voice(who)), "", who)

    def test_a_text_hint_rescues_without_asking_the_voice(self):
        self.assertEqual(mg.rescue_decision(
            "on", True, "no-wake", "Travis, what's new",
            self._voice(mg.NOT_OWNER)), "")
        self.assertEqual(self.asked, 0)

    def test_shadow_only_says_it_would_and_never_asks_the_voice(self):
        for who in (mg.NOT_OWNER, mg.OWNER):
            self.assertEqual(mg.rescue_decision(
                "shadow", True, "no-wake", "la", self._voice(who)),
                "shadow")
        self.assertEqual(self.asked, 0)
        self.assertEqual(mg.rescue_decision(
            "shadow", True, "no-wake", "Travis, hi",
            self._voice(mg.NOT_OWNER)), "")

    def test_errors_rescue(self):
        def boom():
            raise RuntimeError("voice")
        self.assertEqual(mg.rescue_decision("on", True, "no-wake", "la",
                                            boom), "")


class ShadowLostTests(unittest.TestCase):
    """After a shadow rescue: lost = its line passes the wake gates AND
    the voice is not the owner's ('on' rescues OWNER / UNSURE /
    UNAVAILABLE). The voice is asked only for a passing line."""

    def setUp(self):
        self.asked = 0

    def _voice(self, verdict):
        def fn():
            self.asked += 1
            if isinstance(verdict, Exception):
                raise verdict
            return verdict
        return fn

    def test_a_line_that_fails_the_wake_gates_is_never_lost(self):
        self.assertFalse(mg.shadow_lost(False, self._voice(mg.NOT_OWNER)))
        self.assertEqual(self.asked, 0)

    def test_a_wake_line_in_another_voice_is_lost(self):
        self.assertTrue(mg.shadow_lost(True, self._voice(mg.NOT_OWNER)))
        self.assertEqual(self.asked, 1)

    def test_on_would_have_rescued_the_owner(self):
        for who in (mg.OWNER, mg.UNSURE, mg.UNAVAILABLE):
            self.assertFalse(mg.shadow_lost(True, self._voice(who)), who)
        self.assertFalse(mg.shadow_lost(True, None))        # unavailable

    def test_errors_count_nothing(self):
        self.assertFalse(mg.shadow_lost(True,
                                        self._voice(RuntimeError("x"))))


class AmbientTests(unittest.TestCase):
    def test_ambient(self):
        # Over music no ambient batch is transcribed in 'on' — voice-ID
        # cannot pick the owner out of music (10-04: half the lyric batches
        # scored >= 0.45 against his print; his commands over media 0.46-0.52).
        self.assertEqual(mg.ambient_decision("on", True), "skip")
        self.assertEqual(mg.ambient_decision("shadow", True), "shadow")
        self.assertEqual(mg.ambient_decision("on", False), "")
        self.assertEqual(mg.ambient_decision("off", True), "")
        self.assertEqual(mg.ambient_decision("nonsense", True), "")

    def test_there_is_no_capture_cut(self):
        # Review 2026-10-04: no mode shortens a capture (see the module
        # docstring); the counter has no "captures cut" either.
        self.assertFalse(hasattr(mg, "capture_decision"))
        for kind in mg.MinuteCounter.KINDS:
            self.assertNotIn("capture", kind)


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.now = [1000.0]
        self.c = mg.MinuteCounter(clock=lambda: self.now[0])

    def test_no_line_inside_the_minute_or_without_music(self):
        self.c.note("whisper_ambient", 3)
        self.now[0] += 30
        self.assertIsNone(self.c.tick("shadow"))
        self.now[0] += 31
        self.assertIsNone(self.c.tick("shadow"))      # no music: no line
        self.assertEqual(self.c.snapshot()["whisper_ambient"], 0)  # reset

    def test_one_line_per_minute_with_music(self):
        self.c.note("whisper_ambient", 15)
        self.c.note("whisper_turn", 6)
        self.c.note("rescue", 5)
        self.c.note("would_ambient", 15)
        self.c.note("would_rescue", 5)
        self.c.note("lost_rescue", 1)
        self.c.mark_music()
        self.now[0] += 60
        line = self.c.tick("shadow")
        self.assertTrue(line.startswith("[music-gate] shadow: 60 s with music"
                                        " — whisper 21 (ambient 15, turns 6"))
        # "<=": shadow asks no voice before a rescue, so every rescue
        # without a wake hint counts (an upper bound of what 'on' skips).
        self.assertIn("would skip: ambient 15, rescues <=5; lost: rescues 1",
                      line)
        self.assertNotIn("captures cut", line)
        self.assertIsNone(self.c.tick("shadow"))      # a fresh window

    def test_on_says_skipped(self):
        self.c.note("skip_ambient", 14)
        self.c.mark_music()
        self.now[0] += 61
        line = self.c.tick("on")
        self.assertIn("skipped: ambient 14, rescues 0", line)
        self.assertNotIn("would skip", line)
        self.assertNotIn("captures cut", line)

    def test_unknown_kinds_are_ignored(self):
        self.c.note("nonsense", 4)
        self.assertNotIn("nonsense", self.c.snapshot())


if __name__ == "__main__":
    unittest.main()
