"""core.audio_check: "why can't I hear my video?" answered from a READING.

Live 2026-10-09 17:53 the question got an invented "the volume is currently
set to 20%". These pin the recogniser and the answer builder: every number
in the answer comes from the AudioState it was given, an unreadable field is
never guessed, and a command is not a question. Pure (no pycaw), light CI.

    python -m unittest tests.test_audio_check
"""
from __future__ import annotations

import re
import unittest

from core import audio_check as ac
from core import claim_validator as cv

LIVE_USER = "Jarvis, why can't I hear my YouTube video?"


def _state(**kw):
    sessions = kw.pop("sessions", [])
    return ac.AudioState(sessions=sessions, **kw)


class RecogniserTests(unittest.TestCase):
    def test_trouble_questions_match(self):
        for text in (LIVE_USER, "I can't hear anything",
                     "why is there no sound", "is my sound muted",
                     "is the PC muted", "the audio isn't working",
                     "what's the volume at", "there's no audio from Chrome"):
            with self.subTest(text=text):
                self.assertTrue(ac.is_audio_trouble_question(text))

    def test_commands_and_other_talk_do_not(self):
        for text in ("mute it", "turn the volume up",
                     "set the volume to 30 percent", "unmute",
                     "play something on YouTube", "what's the weather", "",
                     None):
            with self.subTest(text=text):
                self.assertFalse(ac.is_audio_trouble_question(text))

    def test_youtube_means_the_browser(self):
        label, frags = ac.target_apps(LIVE_USER)
        self.assertEqual(label, "the browser")
        self.assertIn("chrome", frags)


class DescribeTests(unittest.TestCase):
    def test_master_mute_is_named_as_the_cause(self):
        out = ac.describe(_state(master_pct=40, muted=True,
                                 device="Speakers"), LIVE_USER)
        self.assertIn("muted", out)
        self.assertIn("that would explain it", out)

    def test_app_muted_in_the_mixer(self):
        out = ac.describe(_state(
            master_pct=35, muted=False, device="Speakers",
            sessions=[ac.AppSession("chrome.exe", 100, True, False)]),
            LIVE_USER)
        self.assertIn("Chrome is muted in the volume mixer", out)
        self.assertIn("35 percent", out)

    def test_every_number_comes_from_the_reading(self):
        st = _state(master_pct=35, muted=False, device="Speakers",
                    sessions=[ac.AppSession("chrome.exe", 80, False, True)])
        out = ac.describe(st, LIVE_USER)
        self.assertEqual(sorted(re.findall(r"\d+", out)), ["35", "80"])
        self.assertIn("Speakers", out)
        # ...so the claim validator reads the answer as grounded.
        self.assertIsNone(cv.find_ungrounded_reading(
            "The system volume is at 35 percent, sir.", grounding=[out]))

    def test_no_session_says_the_app_is_not_playing(self):
        out = ac.describe(_state(master_pct=50, muted=False), LIVE_USER)
        self.assertIn("I don't see the browser playing any audio", out)

    def test_silent_session_may_be_paused(self):
        out = ac.describe(_state(
            master_pct=50, muted=False,
            sessions=[ac.AppSession("msedge.exe", 100, False, False)]),
            LIVE_USER)
        self.assertIn("Edge isn't sending any sound", out)

    def test_nothing_readable_invents_nothing(self):
        out = ac.describe(ac.AudioState(), LIVE_USER)
        self.assertIn("couldn't read", out)
        self.assertIsNone(re.search(r"\d", out))

    def test_unreadable_volume_is_not_guessed(self):
        out = ac.describe(_state(muted=False, device="Speakers"), LIVE_USER)
        self.assertIsNone(re.search(r"\d", out))

    def test_ducking_is_mentioned(self):
        out = ac.describe(_state(master_pct=50, muted=False, ducked=True,
                                 sessions=[]), "is it muted")
        self.assertIn("lower other apps while I'm speaking", out)


if __name__ == "__main__":
    unittest.main()
