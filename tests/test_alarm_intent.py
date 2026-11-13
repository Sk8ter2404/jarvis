"""core/alarm_intent.py - a plain "set an alarm" line read without the brain.

Live 2026-11-12: "set an alarm for 422" was passed through to set_alarm by
the brain at 16:21:18 and answered with a question about the number at
17:03:22. These tables pin which lines the matcher takes, what time each
becomes through the REAL skills/alarm parser (fixed clock Thu 2026-11-12
16:21), and which lines it must leave to the brain.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

from core import alarm_intent
from tests._skill_harness import load_skill_isolated

_SAVED_DATA_DIR_ENV = None
_DATA_TMPDIR = None
NOW = datetime(2026, 11, 12, 16, 21, 0)


def setUpModule() -> None:
    global _SAVED_DATA_DIR_ENV, _DATA_TMPDIR
    _SAVED_DATA_DIR_ENV = os.environ.get("JARVIS_DATA_DIR")
    _DATA_TMPDIR = tempfile.mkdtemp(prefix="jarvis_alarm_intent_test_")
    os.environ["JARVIS_DATA_DIR"] = _DATA_TMPDIR


def tearDownModule() -> None:
    if _SAVED_DATA_DIR_ENV is None:
        os.environ.pop("JARVIS_DATA_DIR", None)
    else:
        os.environ["JARVIS_DATA_DIR"] = _SAVED_DATA_DIR_ENV
    if _DATA_TMPDIR:
        shutil.rmtree(_DATA_TMPDIR, ignore_errors=True)


POS = [
    # live lines (time words only)
    ("Jarvis, set an alarm for 422", "Thu 16:22"),
    ("Jarvis setting an alarm for 422. That's really good.", "Thu 16:22"),
    ("Jarvis, set an alarm for one hour.", "Thu 17:21"),
    ("Jarvis, do a test one minute alarm.", "Thu 16:22"),
    ("set an alarm for 5.30 in the morning", "Fri 05:30"),
    # the task's examples
    ("setting an alarm for 4 22", "Thu 16:22"),
    ("wake me up at 6 30", "Thu 18:30"),
    ("alarm in 20 minutes", "Thu 16:41"),
    # spoken clock shapes
    ("set an alarm for 4:22", "Thu 16:22"),
    ("set an alarm for 4.22", "Thu 16:22"),
    ("set an alarm for 1030", "Thu 22:30"),
    ("Set an alarm for 6:30 AM.", "Fri 06:30"),
    ("set an alarm for 6:30 a.m.", "Fri 06:30"),
    ("set an alarm for 6 30 a m", "Fri 06:30"),
    ("set an alarm for 630am", "Fri 06:30"),
    ("set an alarm for 7 pm", "Thu 19:00"),
    ("set an alarm for 7", "Thu 19:00"),
    ("set an alarm for seven", "Thu 19:00"),
    ("set an alarm for seven o'clock", "Thu 19:00"),
    ("set an alarm for six thirty", "Thu 18:30"),
    ("set an alarm for six oh five", "Thu 18:05"),
    ("set an alarm for seven fifteen am", "Fri 07:15"),
    ("set an alarm for 0630", "Fri 06:30"),
    ("set an alarm for 13 30", "Fri 13:30"),
    ("set an alarm for 17:45", "Thu 17:45"),
    ("set an alarm for noon", "Fri 12:00"),
    ("set an alarm for midnight", "Fri 00:00"),
    ("set an alarm for 7 in the morning", "Fri 07:00"),
    ("set an alarm for 9 tonight", "Thu 21:00"),
    ("set an alarm for 6 30 tomorrow morning", "Fri 06:30"),
    ("set an alarm for tomorrow morning at 6 30", "Fri 06:30"),
    ("set an alarm at 5", "Thu 17:00"),
    ("set an alarm for about 7 30", "Thu 19:30"),
    # verbs and leads
    ("Hey Jarvis, can you set an alarm for 6 30 am?", "Fri 06:30"),
    ("Jarvis, could you please set an alarm for 7 am", "Fri 07:00"),
    ("Okay. Jarvis, set an alarm for 8 am, please.", "Fri 08:00"),
    ("Jarvis, I need you to set an alarm for 5 45 am", "Fri 05:45"),
    ("I need an alarm for 6 am", "Fri 06:00"),
    ("I want an alarm at 7 15", "Thu 19:15"),
    ("make an alarm for 8 pm", "Thu 20:00"),
    ("create an alarm for 9 30 pm", "Thu 21:30"),
    ("set up an alarm for 6 am", "Fri 06:00"),
    ("set me an alarm for 6 am", "Fri 06:00"),
    ("set the alarm for 7 am", "Fri 07:00"),
    ("set my alarm for 6 45 am", "Fri 06:45"),
    ("set another alarm for 7 05 am", "Fri 07:05"),
    ("set a wake up alarm for 6 am", "Fri 06:00"),
    ("set a new alarm for 10 pm", "Thu 22:00"),
    ("let's set an alarm for 5 pm", "Thu 17:00"),
    ("I need to set an alarm for 6 am", "Fri 06:00"),
    ("go ahead and set an alarm for 4 30", "Thu 16:30"),
    ("give me an alarm in 45 minutes", "Thu 17:06"),
    ("set an alarm to go off at 6 am", "Fri 06:00"),
    ("set an alarm to go off in 10 minutes", "Thu 16:31"),
    ("set an alarm to wake me up at 6 30 am", "Fri 06:30"),
    ("set an alarm for 7 am, Jarvis", "Fri 07:00"),
    ("set an alarm for 7 am sir. Thanks.", "Fri 07:00"),
    # durations
    ("set an alarm for 20 minutes", "Thu 16:41"),
    ("set an alarm for twenty minutes", "Thu 16:41"),
    ("set an alarm in 20", "Thu 16:41"),
    ("set an alarm for an hour", "Thu 17:21"),
    ("set an alarm for half an hour", "Thu 16:51"),
    ("set an alarm for an hour and a half", "Thu 17:51"),
    ("set an alarm for 2 hours from now", "Thu 18:21"),
    ("set an alarm for 1 hour and 30 minutes", "Thu 17:51"),
    ("set an alarm for 90 seconds", "Thu 16:22"),
    ("set a 20 minute alarm", "Thu 16:41"),
    ("set a 20-minute alarm", "Thu 16:41"),
    ("set a 6 30 am alarm", "Fri 06:30"),
    ("set a 7 am alarm", "Fri 07:00"),
    ("10 minute alarm", "Thu 16:31"),
    ("alarm for 7 am", "Fri 07:00"),
    ("alarm at 6 30", "Thu 18:30"),
    ("an alarm for 5 pm please", "Thu 17:00"),
    # wake me
    ("wake me up at 7", "Thu 19:00"),
    ("wake me up at 7 am", "Fri 07:00"),
    ("wake me at six", "Thu 18:00"),
    ("Jarvis, wake me up at 6:30 in the morning", "Fri 06:30"),
    ("wake me up in an hour", "Thu 17:21"),
    ("wake me up in 20 minutes", "Thu 16:41"),
    ("wake me up in twenty", "Thu 16:41"),
    ("wake me up by 7 am", "Fri 07:00"),
    ("wake me up around 6 am", "Fri 06:00"),
    ("wake me up tomorrow morning at 7", "Fri 07:00"),
    ("wake me up at 5.30 in the morning", "Fri 05:30"),
    ("I need you to wake me up in an hour", "Thu 17:21"),
    ("I'm going to need you to wake me up at 6 am", "Fri 06:00"),
    ("can you wake me up at 6 45 am?", "Fri 06:45"),
    ("get me up at 6 am", "Fri 06:00"),
    ("please wake me up at noon", "Fri 12:00"),
    ("wake me up in half an hour", "Thu 16:51"),
    ("set an alarm for 7 am tomorrow", "Fri 07:00"),
    ("Set an alarm for 6:30 A.M. tomorrow.", "Fri 06:30"),
    ("wake me up tomorrow at 7 am", "Fri 07:00"),
    ("set an alarm for 12 30 am", "Fri 00:30"),
    ("set an alarm for 2 a m", "Fri 02:00"),
    ("wake me up at 10 15", "Thu 22:15"),
    ("set an alarm for 7 30 pm tonight", "Thu 19:30"),
    ("yeah set an alarm for 5 o'clock", "Thu 17:00"),
    ("wake me up at 7 eleven", "Thu 19:11"),
    ("set an alarm for 7 a.m. That's all.", "Fri 07:00"),
    ("Jarvis, alarm, 6 30 am", "Fri 06:30"),
]

NEG = [
    # questions
    "did you set an alarm", "did you set an alarm for 7", "what alarms do I have",
    "do I have an alarm set", "do I have an alarm for 7", "is there an alarm set for 7",
    "what time is my alarm", "when is my alarm going off", "is my alarm set for 6 30",
    "how do I set an alarm for 7", "should I set an alarm for 7", "alarm at 7?",
    "can you tell me if there is an alarm for 7", "what happens if I set an alarm for 7",
    "why did you set an alarm for 11 23", "are you going to wake me up at 7",
    "will you wake me up at 7", "can you list my alarms", "will you set an alarm for 7",
    # negations
    "don't set an alarm", "don't set an alarm for 7", "do not set an alarm for 7",
    "please don't wake me up at 7", "never set an alarm for 7", "no need to set an alarm for 7",
    "I don't need an alarm for 7", "you don't need to wake me up at 7",
    "I didn't want an alarm for 422", "not an alarm for 7",
    # cancel / stop / snooze / list / change
    "cancel the alarm", "cancel my alarm for 7", "cancel the 7 am alarm", "stop the alarm",
    "stop", "snooze", "snooze the alarm for 10 minutes", "turn off the alarm",
    "delete my 6 30 alarm", "remove the alarm for 7", "list my alarms", "show me my alarms",
    "change my alarm to 7", "move the alarm to 7 30", "I'm up", "ten more minutes",
    # past tense / talk about alarms
    "my alarm went off at 7", "the alarm at 7 didn't ring", "I set an alarm for 7",
    "I set an alarm for 7 yesterday", "you set an alarm for 11 23", "the alarm was set for 422",
    "I had an alarm for 6 this morning", "that alarm for 422 was wrong",
    "my alarm for 7 never rang", "alarm at 7 went off", "the alarm for 7 am",
    "I was setting an alarm for 7 when it crashed", "he set an alarm for 7",
    # other alarm senses
    "the smoke alarm went off at 7", "set the smoke alarm for 7", "set the car alarm",
    "arm the security alarm at 7", "the fire alarm is going off", "set off the alarm at 7",
    "alarm system at 7", "false alarm at 7",
    # timers and reminders
    "set a timer for 20 minutes", "set a 20 minute timer", "remind me at 7 to call mom",
    "remind me in 20 minutes", "start a timer for an hour", "set a reminder for 7",
    "set a timer and an alarm for 7", "set an alarm to remind me to take the trash out at 7",
    # someone else / other device
    "tell mom to set an alarm for 7", "wake him up at 7", "wake her up at 6 30",
    "wake the kids up at 7", "wake us up at 7", "set an alarm for 7 on your phone",
    "set an alarm on my phone for 7", "Alexa, set an alarm for 7", "hey Siri set an alarm for 7",
    "she said set an alarm for 7", "can you ask him to wake me up at 7",
    # no time, or a time the grammar must not guess
    "set an alarm", "wake me up", "wake me up in the morning", "set an alarm for tomorrow",
    "wake me up tomorrow", "set an alarm for later", "set an alarm for when the print is done",
    "wake me up before you go", "set an alarm for 20", "set an alarm for 45",
    "set an alarm for 475", "set an alarm for 25 30", "set an alarm for 7 tomorrow",
    "set an alarm for 7 on saturday", "set an alarm for 8 every day", "set an alarm for 3 days",
    "set an alarm for half past six", "set an alarm for quarter to 7", "set an alarm for 10 to 7",
    "set an alarm for 13 pm", "set an alarm for 0 minutes",
    # changes of mind, chains, labels, run-ons
    "set an alarm for 7 never mind", "set an alarm for 7. Actually never mind.",
    "set an alarm for 7. No, 8.", "set an alarm for 7 or 8", "set an alarm for 7 and 7 30",
    "set alarms for 7 and 8", "set an alarm for 7 and turn off the lights",
    "set an alarm for 7 am and play some music", "set an alarm for 7 for work",
    "set an alarm for 7 if it rains", "set an alarm for 5 minutes ago",
    "set an alarm for 7. Good night.", "wake me up at 7 so I can call mom",
    "set an alarm for 7 am. What time is it?",
    # the word in passing
    "I'm going to bed, set an alarm for 7", "maybe set an alarm for 7",
    "if I say set an alarm for 7 what do you do", "the alarm thing", "alarm",
    "is the alarm thing ready yet", "make the alarm loud", "make alarm noises at 7",
    "what does the alarm sound like", "", "Jarvis",
    "set an alarm for 7 30 minutes", "alarm for 7 days", "set an alarm for 7 dollars",
    "set the alarm volume to 50", "set an alarm for 5 people", "set an alarm for 7/15",
    "set an alarm for the 7th", "set an alarm for 7, 30",
    "wake me up tomorrow at 7", "set an alarm for 7 am tomorrow morning tonight",
    "set an alarm for 6 pm in the morning tomorrow", "7 am alarm", "the 7 am alarm",
    "set an alarm for 6 30 that's really good", "set an alarm for 1 2 3",
    "set an alarm for 4 22 22", "set an alarm in 7 am", "wake me up in 7 pm",
    "set an alarm for 99 99", "set an alarm for 24 00", "set an alarm for 7 am 8 am",
    "what if you wake me up at 7", "you could set an alarm for 7", "would you say alarm at 7",
]


class AlarmIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.alarm, _ = load_skill_isolated("alarm", register=False)

    def _when(self, arg):
        with mock.patch.object(self.alarm.time, "time",
                               return_value=NOW.timestamp()):
            secs = self.alarm._parse_when(arg)
        if secs is None:
            return None
        return (NOW + timedelta(seconds=secs)).strftime("%a %H:%M")

    def test_plain_asks_become_the_time_he_said(self):
        for line, want in POS:
            with self.subTest(line=line):
                arg = alarm_intent.match(line)
                self.assertIsNotNone(arg, "not taken")
                self.assertEqual(self._when(arg), want, arg)

    def test_everything_else_is_left_to_the_brain(self):
        for line in NEG:
            with self.subTest(line=line):
                self.assertIsNone(alarm_intent.match(line))

    def test_only_a_reaction_may_follow_the_time(self):
        # An allowlist: no list of bad words can name every way to take an
        # ask back.
        for line in ("set an alarm for 7. Just kidding.",
                     "set an alarm for 7. Hold on.",
                     "set an alarm for 7 am. Different time.",
                     "set an alarm for 7 am. Disregard.",
                     "set an alarm for 7. That's not good.",
                     "set an alarm for 7 am. Psych."):
            with self.subTest(line=line):
                self.assertIsNone(alarm_intent.match(line))
        for line in ("set an alarm for 7 am. Okay thanks.",
                     "set an alarm for 7 am. Perfect.",
                     "Jarvis setting an alarm for 422. That's really good."):
            with self.subTest(line=line):
                self.assertIsNotNone(alarm_intent.match(line))

    def test_it_never_raises(self):
        for bad in (None, 7, b"set an alarm for 7", [], "", "   "):
            self.assertIsNone(alarm_intent.match(bad))


if __name__ == "__main__":
    unittest.main()
